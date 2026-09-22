"""Build cumulative YOLO curriculum datasets from a similarity ranking."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cumulative high-to-low source-domain curricula for YOLOv8."
    )
    parser.add_argument("--source-root", type=Path, default=Path("GC10-DET-2"))
    parser.add_argument(
        "--similarity-csv",
        type=Path,
        default=Path("yolov8_similarity_subsets/source_similarity.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("yolov8_curriculum_subsets")
    )
    parser.add_argument("--source-yaml", type=Path, default=Path("GC10-DET-2/data.yaml"))
    parser.add_argument("--train-images-dir", default="train/images")
    parser.add_argument("--train-labels-dir", default="train/labels")
    parser.add_argument("--val-images-dir", default="valid/images")
    parser.add_argument("--test-images-dir", default="test/images")
    parser.add_argument(
        "--stage-ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.75],
        help="Cumulative fractions of all ranked source samples.",
    )
    parser.add_argument(
        "--build-ablations",
        action="store_true",
        help="Also build top/random 50/75 and bottom-50 source subsets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing generated subset folders."
    )
    return parser.parse_args()


def ratio_tag(ratio: float) -> str:
    percentage = ratio * 100
    return str(int(percentage)) if percentage.is_integer() else f"{percentage:g}"


def validate_stage_ratios(ratios: Sequence[float]) -> List[float]:
    ratios = list(ratios)
    if not ratios:
        raise ValueError("At least one curriculum stage ratio is required.")
    if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError(f"Stage ratios must be in (0, 1], got {ratios}.")
    if any(current <= previous for previous, current in zip(ratios, ratios[1:])):
        raise ValueError(f"Stage ratios must be strictly increasing, got {ratios}.")
    return ratios


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def write_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def load_ranked_paths(csv_path: Path, source_images_root: Path) -> List[Path]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Similarity CSV not found: {csv_path}")
    records: List[Tuple[Path, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("relative_path"):
                image_path = source_images_root / row["relative_path"]
            elif row.get("image_path"):
                image_path = Path(row["image_path"])
            else:
                raise ValueError("CSV needs a relative_path or image_path column.")
            image_path = image_path.resolve()
            try:
                image_path.relative_to(source_images_root)
            except ValueError as exc:
                raise ValueError(f"Ranked image is outside source train root: {image_path}") from exc
            if not image_path.is_file():
                raise FileNotFoundError(f"Ranked source image not found: {image_path}")
            records.append((image_path, float(row["similarity"])))
    if not records:
        raise RuntimeError(f"No ranking records found in: {csv_path}")
    records.sort(key=lambda item: item[1], reverse=True)
    return [path for path, _ in records]


def prepare_subset_root(subset_root: Path, output_root: Path, overwrite: bool) -> None:
    if subset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Generated subset already exists: {subset_root}. Use --overwrite to replace it."
            )
        subset_root.resolve().relative_to(output_root.resolve())
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True)


def copy_selected_train_split(
    selected_paths: Sequence[Path],
    source_images_root: Path,
    source_labels_root: Path,
    subset_root: Path,
) -> int:
    destination_images = subset_root / "train" / "images"
    destination_labels = subset_root / "train" / "labels"
    destination_images.mkdir(parents=True)
    destination_labels.mkdir(parents=True)
    missing_labels = 0

    for image_path in tqdm(selected_paths, desc=f"Copying {subset_root.name}"):
        relative = image_path.relative_to(source_images_root)
        destination_image = destination_images / relative
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, destination_image)

        source_label = (source_labels_root / relative).with_suffix(".txt")
        destination_label = (destination_labels / relative).with_suffix(".txt")
        if source_label.is_file():
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_label, destination_label)
        else:
            missing_labels += 1
    return missing_labels


def make_dataset_yaml(
    source_yaml: dict,
    subset_root: Path,
    source_root: Path,
    val_images_dir: str,
    test_images_dir: str,
) -> dict:
    data = {key: value for key, value in source_yaml.items() if key not in {"path", "train", "val", "test"}}
    data["train"] = (subset_root / "train" / "images").resolve().as_posix()

    val_path = (source_root / val_images_dir).resolve()
    if val_path.is_dir():
        data["val"] = val_path.as_posix()
    else:
        raise FileNotFoundError(f"Source validation images not found: {val_path}")
    test_path = (source_root / test_images_dir).resolve()
    if test_path.is_dir():
        data["test"] = test_path.as_posix()
    return data


def build_subset(
    name: str,
    selected_paths: Sequence[Path],
    source_images_root: Path,
    source_labels_root: Path,
    source_root: Path,
    output_root: Path,
    source_yaml: dict,
    val_images_dir: str,
    test_images_dir: str,
    overwrite: bool,
) -> Path:
    subset_root = output_root / name
    prepare_subset_root(subset_root, output_root, overwrite)
    missing = copy_selected_train_split(
        selected_paths, source_images_root, source_labels_root, subset_root
    )
    dataset_yaml = make_dataset_yaml(
        source_yaml, subset_root, source_root, val_images_dir, test_images_dir
    )
    yaml_path = subset_root / "data.yaml"
    write_yaml(dataset_yaml, yaml_path)
    if missing:
        print(f"[WARN] {name}: {missing} images have no label file (treated as background).")
    print(f"[INFO] Built {name}: {len(selected_paths)} training images")
    return yaml_path.resolve()


def main() -> None:
    args = parse_args()
    ratios = validate_stage_ratios(args.stage_ratios)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_images_root = (source_root / args.train_images_dir).resolve()
    source_labels_root = (source_root / args.train_labels_dir).resolve()
    if not source_images_root.is_dir() or not source_labels_root.is_dir():
        raise FileNotFoundError(
            f"Expected source train images and labels under {source_root}."
        )

    source_yaml = read_yaml(args.source_yaml.resolve())
    ranked_paths = load_ranked_paths(args.similarity_csv.resolve(), source_images_root)
    source_name = source_root.name
    stages: List[Dict] = []
    for index, ratio in enumerate(ratios, start=1):
        count = max(1, int(len(ranked_paths) * ratio))
        tag = ratio_tag(ratio)
        name = f"{source_name}_stage{index}_top{tag}"
        yaml_path = build_subset(
            name=name,
            selected_paths=ranked_paths[:count],
            source_images_root=source_images_root,
            source_labels_root=source_labels_root,
            source_root=source_root,
            output_root=output_root,
            source_yaml=source_yaml,
            val_images_dir=args.val_images_dir,
            test_images_dir=args.test_images_dir,
            overwrite=args.overwrite,
        )
        stages.append(
            {"index": index, "name": name, "ratio": ratio, "samples": count, "data": yaml_path.as_posix()}
        )

    ablations: List[Dict] = []
    if args.build_ablations:
        rng = random.Random(args.seed)
        for kind, ratio in [("top", 0.50), ("top", 0.75), ("bottom", 0.50), ("random", 0.50), ("random", 0.75)]:
            count = max(1, int(len(ranked_paths) * ratio))
            if kind == "top":
                selected = ranked_paths[:count]
            elif kind == "bottom":
                selected = ranked_paths[-count:]
            else:
                selected = rng.sample(ranked_paths, count)
            name = f"{source_name}_{kind}{ratio_tag(ratio)}"
            yaml_path = build_subset(
                name=name,
                selected_paths=selected,
                source_images_root=source_images_root,
                source_labels_root=source_labels_root,
                source_root=source_root,
                output_root=output_root,
                source_yaml=source_yaml,
                val_images_dir=args.val_images_dir,
                test_images_dir=args.test_images_dir,
                overwrite=args.overwrite,
            )
            ablations.append(
                {"name": name, "kind": kind, "ratio": ratio, "samples": count, "data": yaml_path.as_posix()}
            )

    manifest = {
        "source_root": source_root.as_posix(),
        "similarity_csv": args.similarity_csv.resolve().as_posix(),
        "direction": "high-to-low",
        "cumulative": True,
        "total_ranked_samples": len(ranked_paths),
        "stages": stages,
        "ablations": ablations,
    }
    manifest_path = output_root / "curriculum.yaml"
    write_yaml(manifest, manifest_path)
    print(f"[INFO] Curriculum manifest: {manifest_path}")


if __name__ == "__main__":
    main()
