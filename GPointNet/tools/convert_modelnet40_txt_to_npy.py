"""Convert ModelNet40 normal-resampled txt data to GPointNet npy files.

The converter is designed for the PointNet/HiT-ADV style dataset:

    modelnet40_normal_resampled/
      modelnet40_shape_names.txt
      modelnet40_train.txt
      modelnet40_test.txt
      airplane/airplane_0001.txt
      ...

It writes files expected by GPointNet:

    GPointNet/data/airplane_train.npy
    GPointNet/data/airplane_test.npy
    ...

Each output array has shape [num_shapes, num_point, 3]. Normals are ignored.
By default this matches HiT-ADV preprocessing: first 1024 points + per-shape
unit-sphere normalization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def pc_normalize(points: np.ndarray) -> np.ndarray:
    """Normalize one point cloud with the same rule used by HiT-ADV."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    radius = np.max(np.sqrt(np.sum(points * points, axis=1)))
    if radius <= 0:
        raise ValueError("zero-radius point cloud")
    return points / radius


def farthest_point_sample(points: np.ndarray, num_point: int, rng: np.random.Generator) -> np.ndarray:
    """Simple numpy FPS. Keeps all input columns, but samples by xyz distance."""
    count = points.shape[0]
    xyz = points[:, :3]
    centroids = np.empty(num_point, dtype=np.int64)
    distance = np.full(count, 1e10, dtype=np.float64)
    farthest = int(rng.integers(0, count))

    for i in range(num_point):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))

    return points[centroids]


def choose_points(
    points: np.ndarray,
    num_point: int,
    use_fps: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return exactly num_point rows."""
    if points.shape[0] >= num_point:
        if use_fps:
            return farthest_point_sample(points, num_point, rng)
        return points[:num_point]

    pad_size = num_point - points.shape[0]
    pad_idx = rng.choice(points.shape[0], size=pad_size, replace=True)
    return np.concatenate([points, points[pad_idx]], axis=0)


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def class_from_shape_id(shape_id: str) -> str:
    """ModelNet ids look like night_stand_0001; class name is all but suffix."""
    parts = shape_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"unexpected shape id: {shape_id}")
    return "_".join(parts[:-1])


def load_txt_point_cloud(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing point cloud txt: {path}")
    points = np.loadtxt(path, delimiter=",").astype(np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"expected [N, >=3] in {path}, got {points.shape}")
    return points


def validate_modelnet40_root(modelnet_root: Path) -> tuple[list[str], dict[str, list[str]]]:
    class_file = modelnet_root / "modelnet40_shape_names.txt"
    classes = read_lines(class_file)
    if len(classes) != 40:
        raise ValueError(f"expected 40 classes in {class_file}, got {len(classes)}")

    split_ids = {
        "train": read_lines(modelnet_root / "modelnet40_train.txt"),
        "test": read_lines(modelnet_root / "modelnet40_test.txt"),
    }

    missing_class_dirs = [name for name in classes if not (modelnet_root / name).is_dir()]
    if missing_class_dirs:
        raise FileNotFoundError("missing class directories: " + ", ".join(missing_class_dirs))

    missing_txt: list[Path] = []
    for ids in split_ids.values():
        for shape_id in ids:
            class_name = class_from_shape_id(shape_id)
            txt_path = modelnet_root / class_name / f"{shape_id}.txt"
            if not txt_path.is_file():
                missing_txt.append(txt_path)

    if missing_txt:
        preview = "\n".join(str(path) for path in missing_txt[:20])
        suffix = f"\n... and {len(missing_txt) - 20} more" if len(missing_txt) > 20 else ""
        raise FileNotFoundError(f"missing point cloud txt files:\n{preview}{suffix}")

    return classes, split_ids


def convert_split(
    modelnet_root: Path,
    output_dir: Path,
    split: str,
    shape_ids: list[str],
    num_point: int,
    use_fps: bool,
    rng: np.random.Generator,
) -> None:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)

    for idx, shape_id in enumerate(shape_ids, start=1):
        class_name = class_from_shape_id(shape_id)
        txt_path = modelnet_root / class_name / f"{shape_id}.txt"
        points = load_txt_point_cloud(txt_path)
        points = choose_points(points, num_point, use_fps, rng)
        xyz = pc_normalize(points[:, :3]).astype(np.float32)
        grouped[class_name].append(xyz)

        if idx % 500 == 0:
            print(f"[{split}] converted {idx}/{len(shape_ids)} shapes")

    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in sorted(grouped):
        array = np.stack(grouped[class_name], axis=0).astype(np.float32)
        out_path = output_dir / f"{class_name}_{split}.npy"
        np.save(out_path, array)
        print(f"[{split}] {class_name:<16} {array.shape} -> {out_path}")


def parse_args() -> argparse.Namespace:
    gpointnet_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert ModelNet40 txt data into GPointNet data/*.npy files."
    )
    parser.add_argument(
        "--modelnet_root",
        type=Path,
        required=True,
        help="Path to modelnet40_normal_resampled.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=gpointnet_root / "data",
        help="Output directory for {class}_{split}.npy files.",
    )
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
    )
    parser.add_argument(
        "--use_fps",
        action="store_true",
        help="Use FPS instead of the first num_point rows. HiT-ADV default is false.",
    )
    parser.add_argument("--seed", type=int, default=666)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modelnet_root = args.modelnet_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.num_point <= 0:
        raise ValueError("--num_point must be positive")
    if not modelnet_root.is_dir():
        raise FileNotFoundError(f"modelnet root does not exist: {modelnet_root}")

    classes, split_ids = validate_modelnet40_root(modelnet_root)
    print(f"ModelNet root: {modelnet_root}")
    print(f"Output dir:    {output_dir}")
    print(f"Classes:       {len(classes)}")
    print(f"num_point:     {args.num_point}")
    print(f"use_fps:       {args.use_fps}")

    rng = np.random.default_rng(args.seed)
    for split in args.splits:
        convert_split(
            modelnet_root=modelnet_root,
            output_dir=output_dir,
            split=split,
            shape_ids=split_ids[split],
            num_point=args.num_point,
            use_fps=args.use_fps,
            rng=rng,
        )


if __name__ == "__main__":
    main()
