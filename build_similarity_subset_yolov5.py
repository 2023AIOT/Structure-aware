import os
import sys
import csv
import yaml
import math
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# =========================
# 你需要在 YOLOv5 根目录运行
# =========================
from models.experimental import attempt_load


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# 基础工具
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, yaml_path: str):
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"No images found in: {image_dir}")
    return paths


def image_to_label_path(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


# =========================================================
# YOLOv5 backbone 特征提取器
# =========================================================
class YOLOv5BackboneFeatureExtractor(nn.Module):
    """
    默认提取 YOLOv5s 的 backbone 部分（通常是 model.model[:10]）
    对输出做 GAP 得到图像级特征。
    """

    def __init__(self, weight_path: str, device: torch.device):
        super().__init__()
        self.device = device

        ckpt_model = attempt_load(weight_path, device=device)
        ckpt_model.eval()

        print(f"[DEBUG] ckpt_model type: {type(ckpt_model)}")

        # 第一层兼容
        if hasattr(ckpt_model, "model"):
            full_model = ckpt_model.model
        else:
            full_model = ckpt_model

        print(f"[DEBUG] full_model type: {type(full_model)}")

        # 第二层兼容
        if hasattr(full_model, "model"):
            layer_container = full_model.model
        else:
            layer_container = full_model

        print(f"[DEBUG] layer_container type: {type(layer_container)}")
        print(f"[DEBUG] number of layers: {len(layer_container)}")

        # 取前10层作为 backbone
        self.backbone = nn.Sequential(*list(layer_container[:10])).to(device).eval()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def load_image(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


# =========================================================
# 特征提取
# =========================================================
@torch.no_grad()
def extract_features(
    image_paths: List[Path],
    extractor: YOLOv5BackboneFeatureExtractor,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, List[Path]]:
    tensors = []
    valid_paths = []

    for p in tqdm(image_paths, desc="Loading images"):
        try:
            img_tensor = extractor.load_image(p)
            tensors.append(img_tensor)
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    if len(tensors) == 0:
        raise RuntimeError("No valid images were loaded.")

    feats = []
    for i in tqdm(range(0, len(tensors), batch_size), desc="Extracting features"):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        feat = extractor(batch)
        feats.append(feat.cpu())

    feats = torch.cat(feats, dim=0)
    return feats, valid_paths


def compute_similarity_to_target_center(
    source_feats: torch.Tensor,
    target_feats: torch.Tensor
) -> torch.Tensor:
    target_center = target_feats.mean(dim=0, keepdim=True)
    target_center = F.normalize(target_center, p=2, dim=1)
    source_feats = F.normalize(source_feats, p=2, dim=1)
    sims = torch.mm(source_feats, target_center.t()).squeeze(1)
    return sims


# =========================================================
# 复制 YOLO 数据集子集
# =========================================================
def copy_selected_split_with_labels(
    selected_image_paths: List[Path],
    src_images_root: Path,
    src_labels_root: Path,
    dst_images_root: Path,
    dst_labels_root: Path,
):
    ensure_dir(dst_images_root)
    ensure_dir(dst_labels_root)

    missing_labels = 0
    for img_path in tqdm(selected_image_paths, desc=f"Copying {dst_images_root.parent.name}/{dst_images_root.name}"):
        rel = img_path.relative_to(src_images_root)
        dst_img = dst_images_root / rel
        ensure_dir(dst_img.parent)
        shutil.copy2(img_path, dst_img)

        src_label = image_to_label_path(img_path, src_images_root, src_labels_root)
        dst_label = (dst_labels_root / rel).with_suffix(".txt")
        ensure_dir(dst_label.parent)

        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            missing_labels += 1

    if missing_labels > 0:
        print(f"[WARN] Missing labels: {missing_labels}")


def copy_entire_split(src_split_dir: Path, dst_split_dir: Path):
    if dst_split_dir.exists():
        shutil.rmtree(dst_split_dir)
    shutil.copytree(src_split_dir, dst_split_dir)


def build_subset_dataset(
    source_dataset_root: Path,
    output_dataset_root: Path,
    selected_train_images: List[Path],
    source_yaml_path: Path,
    subset_name: str,
):
    """
    生成新的 YOLO 数据集：
    - train：使用筛选后的子集
    - valid/test：原样复制
    - data.yaml：自动生成
    """
    subset_root = output_dataset_root / subset_name
    ensure_dir(subset_root)

    # train
    src_train_images = source_dataset_root / "train" / "images"
    src_train_labels = source_dataset_root / "train" / "labels"
    dst_train_images = subset_root / "train" / "images"
    dst_train_labels = subset_root / "train" / "labels"

    copy_selected_split_with_labels(
        selected_image_paths=selected_train_images,
        src_images_root=src_train_images,
        src_labels_root=src_train_labels,
        dst_images_root=dst_train_images,
        dst_labels_root=dst_train_labels,
    )

    # valid/test 原样复制
    for split in ["valid", "test"]:
        src_split_dir = source_dataset_root / split
        if src_split_dir.exists():
            dst_split_dir = subset_root / split
            copy_entire_split(src_split_dir, dst_split_dir)

    # 读取原始 data.yaml 并生成新 yaml
    yaml_data = read_yaml(str(source_yaml_path))

    # 建议写相对路径，也可改成绝对路径
    yaml_data["train"] = str((subset_root / "train" / "images").as_posix())
    if (subset_root / "valid" / "images").exists():
        yaml_data["val"] = str((subset_root / "valid" / "images").as_posix())
    if (subset_root / "test" / "images").exists():
        yaml_data["test"] = str((subset_root / "test" / "images").as_posix())

    write_yaml(yaml_data, str(subset_root / "data.yaml"))
    print(f"[INFO] Built dataset: {subset_root}")


# =========================================================
# 保存结果
# =========================================================
def save_similarity_csv(records: List[Tuple[str, float]], save_path: Path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "similarity"])
        for p, s in records:
            writer.writerow([p, f"{s:.8f}"])


def save_path_list(paths: List[Path], save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")


# =========================================================
# 主流程
# =========================================================
def main():
    # =====================================================
    # 这里改成你的实际路径
    # =====================================================
    source_dataset_root = Path(r"NEU-DET-2")   # 源域数据集根目录
    target_dataset_root = Path(r"GC10-DET-2")    # 目标域数据集根目录
    yolo_weight_path = r"runs/NEU-DETpre/best.pt"  # GC10-DET 预训练权重
    output_root = Path(r"./NEU_to_GC10_similarity_subsets")        # 输出目录

    batch_size = 16
    seed = 42

    # 比例设置
    top_ratio = 0.75
    low_ratio = 0.25
    random_ratio = 0.75

    # 是否构建子集数据集
    build_subset = True
    # =====================================================

    set_seed(seed)
    ensure_dir(output_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    source_yaml_path = source_dataset_root / "data.yaml"

    # 只对 train 做筛选
    source_train_images_dir = source_dataset_root / "train" / "images"
    target_train_images_dir = target_dataset_root / "train" / "images"

    print("[INFO] Scanning images...")
    source_image_paths = list_images(source_train_images_dir)
    target_image_paths = list_images(target_train_images_dir)

    print(f"[INFO] Source train images: {len(source_image_paths)}")
    print(f"[INFO] Target train images: {len(target_image_paths)}")

    print("[INFO] Building YOLOv5 backbone feature extractor...")
    extractor = YOLOv5BackboneFeatureExtractor(yolo_weight_path, device=device)

    print("[INFO] Extracting target-domain features...")
    target_feats, target_valid_paths = extract_features(
        target_image_paths, extractor, device, batch_size=batch_size
    )

    print("[INFO] Extracting source-domain features...")
    source_feats, source_valid_paths = extract_features(
        source_image_paths, extractor, device, batch_size=batch_size
    )

    print("[INFO] Computing similarity...")
    sims = compute_similarity_to_target_center(source_feats, target_feats)

    records = list(zip(source_valid_paths, sims.tolist()))
    records_sorted = sorted(records, key=lambda x: x[1], reverse=True)

    # 保存排序
    similarity_csv = output_root / "source_similarity.csv"
    save_similarity_csv([(str(p), s) for p, s in records_sorted], similarity_csv)

    n = len(records_sorted)
    top_k = max(1, int(n * top_ratio))
    low_k = max(1, int(n * low_ratio))
    rand_k = max(1, int(n * random_ratio))

    sim_top_paths = [p for p, _ in records_sorted[:top_k]]
    low_bottom_paths = [p for p, _ in records_sorted[-low_k:]]
    all_paths = [p for p, _ in records_sorted]
    random_paths = random.sample(all_paths, rand_k)

    save_path_list(sim_top_paths, output_root / "sim_top75.txt")
    save_path_list(low_bottom_paths, output_root / "low_bottom25.txt")
    save_path_list(random_paths, output_root / "random75.txt")

    print("[INFO] Saved ranking files.")
    print(f"[INFO] Highest similarity sample: {records_sorted[:3]}")
    print(f"[INFO] Lowest similarity sample: {records_sorted[-3:]}")

    if build_subset:
        print("[INFO] Building YOLO subset datasets...")

        build_subset_dataset(
            source_dataset_root=source_dataset_root,
            output_dataset_root=output_root,
            selected_train_images=sim_top_paths,
            source_yaml_path=source_yaml_path,
            subset_name="GC10_sim_top75",
        )

        build_subset_dataset(
            source_dataset_root=source_dataset_root,
            output_dataset_root=output_root,
            selected_train_images=low_bottom_paths,
            source_yaml_path=source_yaml_path,
            subset_name="GC10_low_bottom25",
        )

        build_subset_dataset(
            source_dataset_root=source_dataset_root,
            output_dataset_root=output_root,
            selected_train_images=random_paths,
            source_yaml_path=source_yaml_path,
            subset_name="GC10_random75",
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()