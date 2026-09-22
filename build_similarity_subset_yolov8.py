"""Rank source-domain images with a pretrained YOLOv8 backbone.

The score follows the paper implementation used by the YOLOv5 script:

1. Extract the last-backbone feature map (YOLOv8 layer 9 by default).
2. Apply global average pooling and L2 normalization per image.
3. Average target-domain features into a normalized target prototype.
4. Rank source images by cosine similarity to that prototype.

Run this script from an Ultralytics checkout or an environment where the
``ultralytics`` package is installed.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank source images by YOLOv8 feature similarity to a target domain."
    )
    parser.add_argument("--source-root", type=Path, default=Path("GC10-DET-2"))
    parser.add_argument("--target-root", type=Path, default=Path("NEU-DET-2"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train7/weights/best.pt"),
        help="YOLOv8 checkpoint pretrained on the complete source domain.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("yolov8_similarity_subsets")
    )
    parser.add_argument("--source-train-dir", default="train/images")
    parser.add_argument("--target-train-dir", default="train/images")
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=9,
        help="YOLOv8 model-layer index. Layer 9 is the final backbone SPPF layer.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or a CUDA index such as 0.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.50, 0.75],
        help="Top/random ratios for saved ablation manifests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({value}), but CUDA is unavailable.")
    return device


def validate_ratio(ratio: float) -> None:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"Ratios must be in (0, 1], got {ratio}.")


def list_images(directory: Path) -> List[Path]:
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    images = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )
    if not images:
        raise RuntimeError(f"No supported images found in: {directory}")
    return images


class YOLOv8BackboneFeatureExtractor:
    """Capture one YOLOv8 backbone layer without breaking its graph topology."""

    def __init__(
        self,
        weights: Path,
        device: torch.device,
        feature_layer: int,
        imgsz: int,
        half: bool,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run this script inside the YOLOv8 "
                "repository or install the same version used by the experiment."
            ) from exc

        if not weights.is_file():
            raise FileNotFoundError(f"YOLOv8 checkpoint not found: {weights}")
        if imgsz <= 0:
            raise ValueError("--imgsz must be positive.")

        wrapper = YOLO(str(weights))
        self.network = wrapper.model.to(device).eval()
        self.device = device
        self.imgsz = imgsz
        self.use_half = bool(half and device.type == "cuda")
        if self.use_half:
            self.network.half()

        layers = self.network.model
        layer_index = feature_layer if feature_layer >= 0 else len(layers) + feature_layer
        if not 0 <= layer_index < len(layers):
            raise IndexError(
                f"Feature layer {feature_layer} is invalid for a {len(layers)}-layer model."
            )
        self.layer_index = layer_index
        self._captured: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None
        self._hook = layers[layer_index].register_forward_hook(self._capture_output)

    def _capture_output(self, _module, _inputs, output) -> None:
        self._captured = output

    def close(self) -> None:
        self._hook.remove()

    def load_image(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            image = ImageOps.pad(
                image,
                (self.imgsz, self.imgsz),
                method=resampling.BILINEAR,
                color=(114, 114, 114),
                centering=(0.5, 0.5),
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

    @torch.inference_mode()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        dtype = torch.float16 if self.use_half else torch.float32
        batch = batch.to(device=self.device, dtype=dtype, non_blocking=True)
        self._captured = None
        _ = self.network(batch)
        feature = self._captured
        if feature is None:
            raise RuntimeError(f"Forward hook for layer {self.layer_index} captured no output.")
        if isinstance(feature, (list, tuple)):
            if not feature:
                raise RuntimeError(f"Layer {self.layer_index} returned an empty sequence.")
            feature = feature[-1]
        if not isinstance(feature, torch.Tensor):
            raise TypeError(
                f"Layer {self.layer_index} returned unsupported type {type(feature)!r}."
            )
        if feature.ndim == 4:
            feature = F.adaptive_avg_pool2d(feature.float(), output_size=1).flatten(1)
        elif feature.ndim == 3:
            feature = feature.float().mean(dim=1)
        else:
            feature = feature.float().flatten(1)
        return F.normalize(feature, p=2, dim=1).cpu()


def extract_features(
    image_paths: Sequence[Path],
    extractor: YOLOv8BackboneFeatureExtractor,
    batch_size: int,
    description: str,
) -> Tuple[torch.Tensor, List[Path]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    features: List[torch.Tensor] = []
    valid_paths: List[Path] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=description):
        tensors: List[torch.Tensor] = []
        batch_paths: List[Path] = []
        for path in image_paths[start : start + batch_size]:
            try:
                tensors.append(extractor.load_image(path))
                batch_paths.append(path)
            except (OSError, ValueError) as exc:
                print(f"[WARN] Skipping unreadable image {path}: {exc}")
        if not tensors:
            continue
        features.append(extractor(torch.stack(tensors)))
        valid_paths.extend(batch_paths)

    if not features:
        raise RuntimeError(f"No valid images remained while {description.lower()}.")
    return torch.cat(features, dim=0), valid_paths


def ratio_tag(ratio: float) -> str:
    percentage = ratio * 100
    return str(int(percentage)) if percentage.is_integer() else f"{percentage:g}"


def write_path_list(paths: Iterable[Path], output_path: Path) -> None:
    output_path.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")


def write_rankings(
    records: Sequence[Tuple[Path, float]], source_images_root: Path, output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "relative_path", "image_path", "similarity"])
        for rank, (path, similarity) in enumerate(records, start=1):
            writer.writerow(
                [
                    rank,
                    path.relative_to(source_images_root).as_posix(),
                    path.as_posix(),
                    f"{similarity:.8f}",
                ]
            )


def main() -> None:
    args = parse_args()
    for ratio in args.ratios:
        validate_ratio(ratio)
    set_seed(args.seed)

    source_images_root = (args.source_root / args.source_train_dir).resolve()
    target_images_root = (args.target_root / args.target_train_dir).resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_paths = list_images(source_images_root)
    target_paths = list_images(target_images_root)
    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Source training images: {len(source_paths)}")
    print(f"[INFO] Target training images: {len(target_paths)}")

    extractor = YOLOv8BackboneFeatureExtractor(
        weights=args.weights.resolve(),
        device=device,
        feature_layer=args.feature_layer,
        imgsz=args.imgsz,
        half=args.half,
    )
    try:
        target_features, _ = extract_features(
            target_paths, extractor, args.batch_size, "Target features"
        )
        source_features, valid_source_paths = extract_features(
            source_paths, extractor, args.batch_size, "Source features"
        )
    finally:
        extractor.close()

    target_prototype = F.normalize(target_features.mean(dim=0, keepdim=True), p=2, dim=1)
    similarities = (source_features @ target_prototype.T).squeeze(1)
    records = sorted(
        zip(valid_source_paths, similarities.tolist()), key=lambda item: item[1], reverse=True
    )
    write_rankings(records, source_images_root, output_root / "source_similarity.csv")

    all_ranked_paths = [path for path, _ in records]
    rng = random.Random(args.seed)
    for ratio in sorted(set(args.ratios)):
        count = max(1, int(len(records) * ratio))
        tag = ratio_tag(ratio)
        write_path_list(all_ranked_paths[:count], output_root / f"sim_top{tag}.txt")
        write_path_list(all_ranked_paths[-count:], output_root / f"low_bottom{tag}.txt")
        write_path_list(rng.sample(all_ranked_paths, count), output_root / f"random{tag}.txt")

    print(f"[INFO] Ranking saved to: {output_root / 'source_similarity.csv'}")
    print(f"[INFO] Highest similarities: {records[:3]}")
    print(f"[INFO] Lowest similarities: {records[-3:]}")


if __name__ == "__main__":
    main()
