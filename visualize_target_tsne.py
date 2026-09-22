"""Visualize target-domain object features before and after adaptation.

The script extracts ground-truth ROI features from the same P3/P4/P5 neck
layers in a source-pretrained model and an adapted model. Features from both
conditions are jointly reduced with PCA and t-SNE so that the two panels share
one embedding space.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import normalize

from models.experimental import attempt_load
from utils.augmentations import letterbox


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_COLORS = ("#2878B5", "#E07B39", "#2E8B57", "#C83E4D", "#7A5195", "#7F5539")
DEFAULT_MARKERS = ("o", "s", "^", "D", "P", "X")


def parse_args():
    parser = argparse.ArgumentParser(description="Joint t-SNE of target-domain ROI features")
    parser.add_argument("--before-weights", type=Path, default=Path("runs/GC10-DETpre/best.pt"))
    parser.add_argument("--after-weights", type=Path, default=Path("runs/train/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("NEU-DET-2/data.yaml"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--layers", nargs="+", type=int, default=[17, 20, 23])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/tsne/target_before_after"))
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_config(path):
    path = path.resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [str(names[index]) for index in sorted(names)]
    else:
        names = [str(name) for name in names]
    return path, data, names


def resolve_dataset_path(data_file, value):
    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = Path.cwd() / path
    return repo_path if repo_path.exists() else data_file.parent / path


def labels_dir_from_images(images_dir):
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts)
    raise ValueError(f"Cannot infer labels directory from {images_dir}")


def read_image(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def load_labels(path):
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Invalid label at {path}:{line_number}: {line}")
        cls, x, y, width, height = map(float, fields[:5])
        labels.append((int(cls), x, y, width, height))
    return labels


def collect_samples(images_dir, labels_dir):
    samples = []
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing ground-truth label: {label_path}")
        labels = load_labels(label_path)
        if labels:
            samples.append((image_path, labels))
    if not samples:
        raise RuntimeError(f"No labeled images found in {images_dir}")
    return samples


def preprocess(image, imgsz):
    height, width = image.shape[:2]
    processed, ratio, pad = letterbox(image, new_shape=(imgsz, imgsz), auto=False, stride=32)
    processed = processed[:, :, ::-1].transpose(2, 0, 1)
    processed = np.ascontiguousarray(processed, dtype=np.float32) / 255.0
    return torch.from_numpy(processed), (height, width), ratio, pad


def transform_box_to_input(label, original_shape, ratio, pad, imgsz):
    cls, x, y, width, height = label
    original_height, original_width = original_shape
    x1 = (x - width / 2) * original_width
    y1 = (y - height / 2) * original_height
    x2 = (x + width / 2) * original_width
    y2 = (y + height / 2) * original_height
    x1, x2 = x1 * ratio[0] + pad[0], x2 * ratio[0] + pad[0]
    y1, y2 = y1 * ratio[1] + pad[1], y2 * ratio[1] + pad[1]
    return cls, np.array([x1 / imgsz, y1 / imgsz, x2 / imgsz, y2 / imgsz], dtype=np.float32).clip(0, 1)


def roi_average(feature_map, batch_index, normalized_box):
    _, _, height, width = feature_map.shape
    x1, y1, x2, y2 = normalized_box
    left = min(width - 1, max(0, int(math.floor(x1 * width))))
    top = min(height - 1, max(0, int(math.floor(y1 * height))))
    right = min(width, max(left + 1, int(math.ceil(x2 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(y2 * height))))
    crop = feature_map[batch_index:batch_index + 1, :, top:bottom, left:right]
    return F.adaptive_avg_pool2d(crop, output_size=1).flatten()


@torch.no_grad()
def extract_object_features(weights, samples, layers, imgsz, batch_size, device):
    model = attempt_load(str(weights), device=device, inplace=True, fuse=True)
    model = model.float().eval()
    if not hasattr(model, "model"):
        raise TypeError(f"Expected a YOLOv5 model in {weights}, got {type(model)}")
    if max(layers) >= len(model.model):
        raise ValueError(f"Layer index {max(layers)} exceeds model length {len(model.model)}")

    activations = {}
    hooks = []
    for layer_index in layers:
        def save_output(_module, _inputs, output, index=layer_index):
            # Detect mutates its input list in place, so preserve neck tensors here.
            activations[index] = output.detach().clone()

        hooks.append(model.model[layer_index].register_forward_hook(save_output))

    all_features, all_classes, all_ids = [], [], []
    try:
        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start:batch_start + batch_size]
            tensors, transformed_labels = [], []
            for image_path, labels in batch_samples:
                image = read_image(image_path)
                tensor, shape, ratio, pad = preprocess(image, imgsz)
                tensors.append(tensor)
                transformed_labels.append([
                    transform_box_to_input(label, shape, ratio, pad, imgsz) for label in labels
                ])

            batch = torch.stack(tensors).to(device, non_blocking=True)
            activations.clear()
            model(batch)
            missing_layers = [index for index in layers if index not in activations]
            if missing_layers:
                raise RuntimeError(f"No activation captured for layers: {missing_layers}")

            for batch_index, ((image_path, _labels), boxes) in enumerate(zip(batch_samples, transformed_labels)):
                for object_index, (cls, box) in enumerate(boxes):
                    pooled = [roi_average(activations[index], batch_index, box) for index in layers]
                    all_features.append(torch.cat(pooled).cpu().numpy())
                    all_classes.append(cls)
                    all_ids.append(f"{image_path.stem}:{object_index}")
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.asarray(all_features, dtype=np.float32), np.asarray(all_classes), all_ids


def cluster_metrics(features, labels):
    centroids = np.stack([features[labels == cls].mean(axis=0) for cls in np.unique(labels)])
    within_distances = []
    for cls, centroid in zip(np.unique(labels), centroids):
        within_distances.extend(np.linalg.norm(features[labels == cls] - centroid, axis=1))
    pairwise = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    within = float(np.mean(within_distances))
    between = float(np.mean(upper))
    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
        "mean_intra_class_distance": within,
        "mean_inter_class_centroid_distance": between,
        "inter_intra_ratio": between / max(within, 1e-12),
    }


def plot_embedding(embedding, labels, names, metrics, output_dir):
    count = len(labels)
    before_embedding, after_embedding = embedding[:count], embedding[count:]
    display_names = [name.replace("_", " ").capitalize() for name in names]
    colors = [DEFAULT_COLORS[index % len(DEFAULT_COLORS)] for index in range(len(names))]
    markers = [DEFAULT_MARKERS[index % len(DEFAULT_MARKERS)] for index in range(len(names))]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    titles = ("(a) Before adaptation", "(b) After adaptation")
    conditions = ((before_embedding, metrics["before"]), (after_embedding, metrics["after"]))

    x_min, y_min = embedding.min(axis=0)
    x_max, y_max = embedding.max(axis=0)
    x_pad = max((x_max - x_min) * 0.04, 1.0)
    y_pad = max((y_max - y_min) * 0.04, 1.0)

    for axis, title, (points, condition_metrics) in zip(axes, titles, conditions):
        for class_index, class_name in enumerate(display_names):
            mask = labels == class_index
            axis.scatter(
                points[mask, 0], points[mask, 1], s=28, marker=markers[class_index],
                c=colors[class_index], alpha=0.76, linewidths=0.35, edgecolors="white",
                label=class_name, rasterized=True,
            )
            centroid = points[mask].mean(axis=0)
            axis.scatter(
                centroid[0], centroid[1], s=105, marker=markers[class_index],
                c=colors[class_index], linewidths=1.1, edgecolors="black", zorder=5,
            )
        axis.set_title(title, fontweight="bold", pad=10)
        axis.set_xlabel("t-SNE dimension 1")
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.65)
        axis.set_axisbelow(True)
        axis.text(
            0.025, 0.975,
            f"Silhouette = {condition_metrics['silhouette_score']:.3f}\n"
            f"DBI = {condition_metrics['davies_bouldin_index']:.3f}",
            transform=axis.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
        )
    axes[0].set_ylabel("t-SNE dimension 2")

    handles = [
        Line2D([0], [0], marker=markers[index], linestyle="", markersize=7.5,
               markerfacecolor=colors[index], markeredgecolor="white", label=name)
        for index, name in enumerate(display_names)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), columnspacing=1.8, handletextpad=0.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.19, wspace=0.08)
    fig.savefig(output_dir / "target_feature_tsne.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "target_feature_tsne.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_outputs(output_dir, embedding, labels, object_ids, names, metrics, config):
    count = len(labels)
    with (output_dir / "tsne_embedding.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "object_id", "class_id", "class_name", "tsne_1", "tsne_2"])
        for condition, offset in (("before", 0), ("after", count)):
            for index, (object_id, cls) in enumerate(zip(object_ids, labels)):
                point = embedding[offset + index]
                writer.writerow([condition, object_id, int(cls), names[int(cls)], point[0], point[1]])

    with (output_dir / "cluster_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", *next(iter(metrics.values())).keys()])
        for condition, values in metrics.items():
            writer.writerow([condition, *[f"{value:.8f}" for value in values.values()]])

    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    data_file, data, names = load_data_config(args.data)
    images_dir = resolve_dataset_path(data_file, data[args.split]).resolve()
    labels_dir = labels_dir_from_images(images_dir)
    samples = collect_samples(images_dir, labels_dir)

    print(f"Device: {device}")
    print(f"Target images: {len(samples)}")
    print(f"Extracting before-adaptation features from {args.before_weights}...")
    before, labels_before, ids_before = extract_object_features(
        args.before_weights, samples, args.layers, args.imgsz, args.batch_size, device
    )
    print(f"Extracting after-adaptation features from {args.after_weights}...")
    after, labels_after, ids_after = extract_object_features(
        args.after_weights, samples, args.layers, args.imgsz, args.batch_size, device
    )
    if ids_before != ids_after or not np.array_equal(labels_before, labels_after):
        raise RuntimeError("Before/after features do not correspond to the same target objects")
    if before.shape != after.shape:
        raise RuntimeError(f"Feature shapes differ: before={before.shape}, after={after.shape}")

    joint = normalize(np.concatenate([before, after], axis=0), norm="l2")
    pca_components = min(args.pca_components, joint.shape[0] - 1, joint.shape[1])
    pca = PCA(n_components=pca_components, random_state=args.seed)
    reduced = pca.fit_transform(joint)
    count = len(labels_before)
    metrics = {
        "before": cluster_metrics(reduced[:count], labels_before),
        "after": cluster_metrics(reduced[count:], labels_before),
    }

    perplexity = min(args.perplexity, (joint.shape[0] - 1) / 3)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        n_iter=args.iterations,
        random_state=args.seed,
        method="barnes_hut",
        angle=0.5,
    )
    embedding = tsne.fit_transform(reduced)

    config = {
        "before_weights": str(args.before_weights.resolve()),
        "after_weights": str(args.after_weights.resolve()),
        "data": str(data_file),
        "split": args.split,
        "images": len(samples),
        "objects_per_condition": int(count),
        "class_counts": {names[index]: int(np.sum(labels_before == index)) for index in range(len(names))},
        "layers": args.layers,
        "imgsz": args.imgsz,
        "pca_components": pca_components,
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "tsne_perplexity": perplexity,
        "tsne_iterations": args.iterations,
        "random_seed": args.seed,
        "joint_embedding": True,
    }
    save_outputs(output_dir, embedding, labels_before, ids_before, names, metrics, config)
    plot_embedding(embedding, labels_before, names, metrics, output_dir)

    print(f"Objects per condition: {count}")
    print(f"Feature dimension: {before.shape[1]}")
    print(f"PCA explained variance: {config['pca_explained_variance_ratio']:.4f}")
    for condition, values in metrics.items():
        formatted = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
        print(f"{condition}: {formatted}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
