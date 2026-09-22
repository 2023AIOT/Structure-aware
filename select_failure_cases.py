"""Rank and visualize object-detection failure cases from YOLO label files.

The prediction files must use YOLO's ``class x y w h confidence`` format,
which can be produced by detect.py with ``--save-txt --save-conf``. Ground
truth files use the standard ``class x y w h`` format.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Select and visualize YOLO failure cases")
    parser.add_argument("--data", type=Path, default=Path("NEU-DET-2/data.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--pred-labels", type=Path, required=True,
                        help="directory containing predicted YOLO txt files with confidence")
    parser.add_argument("--output", type=Path, default=Path("runs/failure_cases/selected"))
    parser.add_argument("--conf-thres", type=float, default=0.25,
                        help="prediction confidence used for failure analysis")
    parser.add_argument("--match-iou", type=float, default=0.50,
                        help="IoU at which a prediction is considered geometrically matched")
    parser.add_argument("--localization-iou", type=float, default=0.10,
                        help="minimum overlap used to identify a localization-error candidate")
    parser.add_argument("--per-type", type=int, default=3,
                        help="maximum selected examples for each failure type")
    parser.add_argument("--images", nargs="*", default=None,
                        help="optional image names/stems to render instead of automatic selection")
    parser.add_argument("--panel-size", type=int, default=600,
                        help="width and height of each panel in the output triptych")
    return parser.parse_args()


def resolve_dataset_path(data_file, value):
    path = Path(value)
    if path.is_absolute():
        return path
    # YOLOv5 data files in this repository use paths relative to the repo root.
    repo_candidate = Path.cwd() / path
    return repo_candidate if repo_candidate.exists() else data_file.parent / path


def labels_dir_from_images(images_dir):
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts)
    raise ValueError(f"Cannot infer labels directory from image directory: {images_dir}")


def load_names(data):
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def load_yolo_labels(path, require_confidence=False):
    rows = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        minimum = 6 if require_confidence else 5
        if len(fields) < minimum:
            raise ValueError(f"Invalid label at {path}:{line_number}: {line}")
        cls = int(float(fields[0]))
        x, y, width, height = map(float, fields[1:5])
        confidence = float(fields[5]) if len(fields) >= 6 else 1.0
        rows.append({
            "cls": cls,
            "xyxy": np.array([
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
            ], dtype=np.float32).clip(0.0, 1.0),
            "conf": confidence,
        })
    return rows


def box_iou_matrix(left, right):
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    a = np.stack([item["xyxy"] for item in left])
    b = np.stack([item["xyxy"] for item in right])
    intersection_min = np.maximum(a[:, None, :2], b[None, :, :2])
    intersection_max = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.clip(intersection_max - intersection_min, 0.0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0.0, None).prod(axis=1)[:, None]
    area_b = np.clip(b[:, 2:] - b[:, :2], 0.0, None).prod(axis=1)[None, :]
    return intersection / np.clip(area_a + area_b - intersection, 1e-9, None)


def greedy_pairs(iou, minimum_iou, allowed=None):
    candidates = []
    for gt_index in range(iou.shape[0]):
        for pred_index in range(iou.shape[1]):
            if iou[gt_index, pred_index] >= minimum_iou:
                if allowed is None or allowed(gt_index, pred_index):
                    candidates.append((float(iou[gt_index, pred_index]), gt_index, pred_index))
    candidates.sort(reverse=True)
    used_gt, used_pred, pairs = set(), set(), []
    for overlap, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_pred:
            pairs.append((gt_index, pred_index, overlap))
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    return pairs


def analyse_image(gt, predictions, match_iou, localization_iou):
    iou = box_iou_matrix(gt, predictions)
    geometric_pairs = greedy_pairs(iou, match_iou)
    correct_pairs = [pair for pair in geometric_pairs if gt[pair[0]]["cls"] == predictions[pair[1]]["cls"]]
    wrong_pairs = [pair for pair in geometric_pairs if gt[pair[0]]["cls"] != predictions[pair[1]]["cls"]]
    correct_gt = {pair[0] for pair in correct_pairs}
    correct_pred = {pair[1] for pair in correct_pairs}

    # A localization error is a same-class prediction that overlaps the target,
    # but not enough to satisfy the evaluation IoU threshold.
    localization_pairs = greedy_pairs(
        iou,
        localization_iou,
        allowed=lambda g, p: (
            iou[g, p] < match_iou
            and gt[g]["cls"] == predictions[p]["cls"]
            and g not in correct_gt
            and p not in correct_pred
        ),
    )

    fn_count = len(gt) - len(correct_pairs)
    fp_count = len(predictions) - len(correct_pairs)
    mean_matched_iou = float(np.mean([pair[2] for pair in correct_pairs])) if correct_pairs else 0.0
    score = (
        8.0 * len(wrong_pairs)
        + 5.0 * len(localization_pairs)
        + 3.0 * fn_count
        + 2.0 * fp_count
        + (1.0 - mean_matched_iou if correct_pairs else 1.0)
    )
    failure_types = []
    if wrong_pairs:
        failure_types.append("wrong_class")
    if fn_count:
        failure_types.append("missed_detection")
    if fp_count:
        failure_types.append("false_positive")
    if localization_pairs:
        failure_types.append("localization")
    return {
        "gt_count": len(gt),
        "pred_count": len(predictions),
        "tp": len(correct_pairs),
        "fn": fn_count,
        "fp": fp_count,
        "wrong_class": len(wrong_pairs),
        "localization": len(localization_pairs),
        "mean_matched_iou": mean_matched_iou,
        "score": score,
        "failure_types": failure_types,
        "correct_pairs": correct_pairs,
        "wrong_pairs": wrong_pairs,
        "localization_pairs": localization_pairs,
    }


def read_image(path):
    # np.fromfile/imdecode supports non-ASCII Windows paths more reliably than cv2.imread.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def write_image(path, image):
    suffix = path.suffix or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Unable to encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def draw_boxes(image, boxes, names, color, include_confidence):
    height, width = image.shape[:2]
    line_width = max(1, round((height + width) / 500))
    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        pixel_box = (
            int(round(x1 * width)), int(round(y1 * height)),
            int(round(x2 * width)), int(round(y2 * height)),
        )
        cv2.rectangle(image, pixel_box[:2], pixel_box[2:], color, line_width, cv2.LINE_AA)
        class_name = names[item["cls"]] if item["cls"] < len(names) else str(item["cls"])
        label = f"{class_name} {item['conf']:.2f}" if include_confidence else class_name
        font_scale = max(0.25, min(width, height) / 1000)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        text_x = max(0, pixel_box[0])
        text_y = max(text_height + baseline + 2, pixel_box[1])
        cv2.rectangle(
            image,
            (text_x, text_y - text_height - baseline - 3),
            (min(width - 1, text_x + text_width + 4), text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            image, label, (text_x + 2, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), line_width, cv2.LINE_AA,
        )
    return image


def make_panel(image, title, size):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    header_height = 58
    panel = np.full((size + header_height, size, 3), 255, dtype=np.uint8)
    panel[header_height:] = resized
    font_scale = 0.85
    text_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
    if text_width > size - 32:
        font_scale = max(0.42, font_scale * (size - 32) / text_width)
    cv2.putText(panel, title, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (20, 20, 20), 2, cv2.LINE_AA)
    return panel


def render_triptych(record, names, output_path, panel_size):
    original = read_image(record["image_path"])
    gt_view = draw_boxes(original.copy(), record["gt"], names, (40, 180, 40), False)
    pred_view = draw_boxes(original.copy(), record["predictions"], names, (30, 30, 220), True)
    metric_text = (
        f"Prediction  TP={record['tp']} FN={record['fn']} FP={record['fp']} "
        f"Wrong={record['wrong_class']} Loc={record['localization']}"
    )
    triptych = np.concatenate([
        make_panel(original, "Original", panel_size),
        make_panel(gt_view, "Ground Truth (GT)", panel_size),
        make_panel(pred_view, metric_text, panel_size),
    ], axis=1)
    write_image(output_path, triptych)


def choose_candidates(records, per_type):
    priorities = {
        "wrong_class": lambda row: (row["wrong_class"], row["score"]),
        "localization": lambda row: (row["localization"], row["score"]),
        "missed_detection": lambda row: (row["fn"], row["score"]),
        "false_positive": lambda row: (row["fp"], row["score"]),
    }
    selected = []
    selected_stems = set()
    for failure_type, key in priorities.items():
        candidates = [row for row in records if failure_type in row["failure_types"]]
        candidates.sort(key=key, reverse=True)
        count = 0
        for row in candidates:
            if row["stem"] in selected_stems:
                continue
            selected.append((failure_type, row))
            selected_stems.add(row["stem"])
            count += 1
            if count >= per_type:
                break
    return selected


def main():
    args = parse_args()
    data_file = args.data.resolve()
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    names = load_names(data)
    images_dir = resolve_dataset_path(data_file, data[args.split]).resolve()
    gt_labels_dir = labels_dir_from_images(images_dir)
    pred_labels_dir = args.pred_labels.resolve()
    output_dir = args.output.resolve()
    panels_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    records = []
    for image_path in image_paths:
        gt = load_yolo_labels(gt_labels_dir / f"{image_path.stem}.txt")
        predictions = [
            item for item in load_yolo_labels(
                pred_labels_dir / f"{image_path.stem}.txt", require_confidence=True
            ) if item["conf"] >= args.conf_thres
        ]
        result = analyse_image(gt, predictions, args.match_iou, args.localization_iou)
        gt_classes = sorted({item["cls"] for item in gt})
        record = {
            "stem": image_path.stem,
            "image_path": image_path,
            "gt": gt,
            "predictions": predictions,
            "gt_classes": ";".join(names[index] for index in gt_classes),
            **result,
        }
        records.append(record)

    records.sort(key=lambda row: row["score"], reverse=True)
    csv_path = output_dir / "failure_case_ranking.csv"
    columns = [
        "rank", "image", "gt_classes", "failure_types", "gt_count", "pred_count",
        "tp", "fn", "fp", "wrong_class", "localization", "mean_matched_iou", "score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(records, start=1):
            writer.writerow({
                "rank": rank,
                "image": row["image_path"].name,
                "gt_classes": row["gt_classes"],
                "failure_types": ";".join(row["failure_types"]) or "none",
                "gt_count": row["gt_count"],
                "pred_count": row["pred_count"],
                "tp": row["tp"],
                "fn": row["fn"],
                "fp": row["fp"],
                "wrong_class": row["wrong_class"],
                "localization": row["localization"],
                "mean_matched_iou": f"{row['mean_matched_iou']:.4f}",
                "score": f"{row['score']:.4f}",
            })

    if args.images:
        requested_stems = [Path(name).stem for name in args.images]
        record_by_stem = {row["stem"]: row for row in records}
        missing = [stem for stem in requested_stems if stem not in record_by_stem]
        if missing:
            raise ValueError(f"Requested images are not in the {args.split} split: {missing}")
        selected = []
        manual_priority = ("wrong_class", "localization", "missed_detection", "false_positive")
        for stem in requested_stems:
            row = record_by_stem[stem]
            primary_type = next(
                (failure_type for failure_type in manual_priority if failure_type in row["failure_types"]),
                "none",
            )
            selected.append((primary_type, row))
    else:
        selected = choose_candidates(records, args.per_type)
    selected_csv_path = output_dir / "selected_candidates.csv"
    with selected_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_order", "primary_failure_type", "image", "gt_classes", "panel"])
        rendered_panels = []
        for order, (failure_type, row) in enumerate(selected, start=1):
            panel_name = f"{order:02d}_{failure_type}_{row['stem']}.jpg"
            panel_path = panels_dir / panel_name
            render_triptych(row, names, panel_path, args.panel_size)
            rendered_panels.append(panel_path)
            writer.writerow([order, failure_type, row["image_path"].name, row["gt_classes"], str(panel_path)])

    if rendered_panels:
        separator = np.full((12, args.panel_size * 3, 3), 255, dtype=np.uint8)
        composite_parts = []
        for index, panel_path in enumerate(rendered_panels):
            if index:
                composite_parts.append(separator)
            composite_parts.append(read_image(panel_path))
        write_image(output_dir / "failure_cases_composite.jpg", np.vstack(composite_parts))

    failures = sum(bool(row["failure_types"]) for row in records)
    print(f"Analysed {len(records)} images; {failures} contain at least one failure at conf={args.conf_thres}.")
    print(f"Full ranking: {csv_path}")
    print(f"Selected candidates: {selected_csv_path}")
    print(f"Triptych panels: {panels_dir}")
    print(f"Composite figure: {output_dir / 'failure_cases_composite.jpg'}")


if __name__ == "__main__":
    main()
