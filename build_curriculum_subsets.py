import csv
import shutil
from pathlib import Path
from typing import List, Tuple
import yaml
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, yaml_path: Path):
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def image_to_label_path(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


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
    for img_path in tqdm(selected_image_paths, desc=f"Copying -> {dst_images_root.parent.name}/{dst_images_root.name}"):
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
    if not src_split_dir.exists():
        return
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
    subset_root = output_dataset_root / subset_name
    ensure_dir(subset_root)

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

    for split in ["valid", "test"]:
        copy_entire_split(source_dataset_root / split, subset_root / split)

    yaml_data = read_yaml(source_yaml_path)
    yaml_data["train"] = str((subset_root / "train" / "images").as_posix())
    if (subset_root / "valid" / "images").exists():
        yaml_data["val"] = str((subset_root / "valid" / "images").as_posix())
    if (subset_root / "test" / "images").exists():
        yaml_data["test"] = str((subset_root / "test" / "images").as_posix())

    write_yaml(yaml_data, subset_root / "data.yaml")
    print(f"[INFO] Built dataset: {subset_root}")


def load_similarity_csv(csv_path: Path) -> List[Tuple[Path, float]]:
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = Path(row["image_path"])
            sim = float(row["similarity"])
            records.append((img_path, sim))
    records.sort(key=lambda x: x[1], reverse=True)
    return records


def main():
    # ===== 修改这里 =====
    source_dataset_root = Path("GC10-DET-2")
    output_root = Path("similarity_subsets")
    similarity_csv = output_root / "source_similarity.csv"
    source_yaml_path = source_dataset_root / "data.yaml"
    # ====================

    records = load_similarity_csv(similarity_csv)
    n = len(records)

    high_end = int(n * 0.25)
    mid_end = int(n * 0.75)

    high25 = [p for p, _ in records[:high_end]]
    mid50 = [p for p, _ in records[high_end:mid_end]]
    low25 = [p for p, _ in records[mid_end:]]

    top75 = [p for p, _ in records[:mid_end]]
    low75 = [p for p, _ in records[high_end:]]

    print(f"[INFO] Total samples: {n}")
    print(f"[INFO] High25: {len(high25)}")
    print(f"[INFO] Mid50 : {len(mid50)}")
    print(f"[INFO] Low25 : {len(low25)}")
    print(f"[INFO] Top75 : {len(top75)}")
    print(f"[INFO] Low75 : {len(low75)}")

    build_subset_dataset(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_root,
        selected_train_images=high25,
        source_yaml_path=source_yaml_path,
        subset_name="GC10_high25",
    )

    build_subset_dataset(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_root,
        selected_train_images=mid50,
        source_yaml_path=source_yaml_path,
        subset_name="GC10_mid50",
    )

    build_subset_dataset(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_root,
        selected_train_images=low25,
        source_yaml_path=source_yaml_path,
        subset_name="GC10_low25",
    )

    build_subset_dataset(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_root,
        selected_train_images=top75,
        source_yaml_path=source_yaml_path,
        subset_name="GC10_top75",
    )

    build_subset_dataset(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_root,
        selected_train_images=low75,
        source_yaml_path=source_yaml_path,
        subset_name="GC10_low75",
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()