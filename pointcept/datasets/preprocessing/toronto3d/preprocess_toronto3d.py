"""
Preprocessing Script for Toronto3D

Converts the raw Toronto3D `.las` tiles (L001..L004) into the per-scene `.npy`
folder format expected by Pointcept's DefaultDataset.

Toronto3D (LAS 1.4, point format 7) per-point attributes used:
  - coord     : exact_x/exact_y/exact_z (double) -> centered float32 (UTM offset removed)
  - color     : 16-bit red/green/blue     -> float32 in [0, 255]
  - strength  : intensity (uint16)         -> float32 normalized to ~[0, 1]
  - segment   : extra-byte `label` (uint8) -> int32, remapped 1..8 -> 0..7, 0 -> ignore_index

Standard benchmark split: train = L001/L003/L004, val = test = L002.

Author: Benchmarking setup for Pointcept
"""

import argparse
import glob
import os

import numpy as np

try:
    import laspy
except ImportError as e:
    raise ImportError(
        "laspy is required. Install with `pip install 'laspy[lazrs]'`."
    ) from e

# Standard Toronto3D split (tile stem -> Pointcept split folder).
# L002 is the held-out test tile in the official benchmark.
SPLIT_MAP = {
    "L001": "train",
    "L002": "val",  # also mirrored into `test`
    "L003": "train",
    "L004": "train",
}

# Toronto3D raw label -> training class id.
#   0 = Unclassified  -> ignore_index
#   1 = Road          -> 0
#   2 = Road marking  -> 1
#   3 = Natural       -> 2
#   4 = Building       -> 3
#   5 = Utility line   -> 4
#   6 = Pole           -> 5
#   7 = Car            -> 6
#   8 = Fence          -> 7
IGNORE_INDEX = -1


def get_field(las, *names):
    """Return the first available point dimension among `names`, else None."""
    available = set(las.point_format.dimension_names)
    for n in names:
        if n in available:
            return np.asarray(las[n])
    return None


def process_tile(las_path, split_root, ignore_index=IGNORE_INDEX):
    stem = os.path.splitext(os.path.basename(las_path))[0]  # e.g. "L001_6cm"
    tile_id = stem.split("_")[0]  # e.g. "L001"
    split = SPLIT_MAP.get(tile_id)
    if split is None:
        print(f"[skip] {stem}: tile id {tile_id} not in SPLIT_MAP")
        return

    print(f"[read] {las_path}")
    las = laspy.read(las_path)
    n = len(las.points)

    # --- coordinates: prefer high-precision exact_* extra dims, fall back to scaled xyz ---
    ex = get_field(las, "exact_x")
    ey = get_field(las, "exact_y")
    ez = get_field(las, "exact_z")
    if ex is not None and ey is not None and ez is not None:
        coord = np.stack([ex, ey, ez], axis=1).astype(np.float64)
    else:
        coord = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    # remove per-tile min offset before casting to float32 (UTM values ~1e6 lose cm precision otherwise)
    coord = coord - coord.min(axis=0, keepdims=True)
    coord = coord.astype(np.float32)

    # --- color: 16-bit -> [0, 255] float ---
    r = get_field(las, "red")
    g = get_field(las, "green")
    b = get_field(las, "blue")
    if r is not None:
        color = np.stack([r, g, b], axis=1).astype(np.float32)
        if color.max() > 255:  # 16-bit color -> 8-bit range
            color = color / 256.0
    else:
        color = np.zeros((n, 3), dtype=np.float32)

    # --- strength: intensity normalized to ~[0, 1] (per-tile) ---
    intensity = get_field(las, "intensity")
    if intensity is not None:
        intensity = intensity.astype(np.float32)
        imax = intensity.max()
        strength = (intensity / imax) if imax > 0 else intensity
        strength = strength.reshape(-1, 1).astype(np.float32)
    else:
        strength = np.zeros((n, 1), dtype=np.float32)

    # --- segment: label extra dim -> 0..7, 0 -> ignore ---
    label = get_field(las, "label", "Label", "scalar_Label", "classification")
    if label is None:
        raise RuntimeError(f"{stem}: no label field found in {list(las.point_format.dimension_names)}")
    label = label.astype(np.int64)
    segment = np.where(label >= 1, label - 1, ignore_index).astype(np.int32)

    out_dir = os.path.join(split_root, split, tile_id)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "coord.npy"), coord)
    np.save(os.path.join(out_dir, "color.npy"), color)
    np.save(os.path.join(out_dir, "strength.npy"), strength)
    np.save(os.path.join(out_dir, "segment.npy"), segment)

    uniq, cnts = np.unique(segment, return_counts=True)
    print(
        f"[done] {tile_id} -> {split}: {n} pts | "
        f"coord[min={coord.min(axis=0)}, max={coord.max(axis=0)}] | "
        f"color[{color.min():.0f}-{color.max():.0f}] | "
        f"strength[{strength.min():.3f}-{strength.max():.3f}] | "
        f"labels={dict(zip(uniq.tolist(), cnts.tolist()))}"
    )

    # mirror L002 (val) into a `test` split too, so scripts/test.sh can evaluate it
    if split == "val":
        test_dir = os.path.join(split_root, "test", tile_id)
        os.makedirs(test_dir, exist_ok=True)
        for asset in ("coord", "color", "strength", "segment"):
            src = os.path.join(out_dir, f"{asset}.npy")
            dst = os.path.join(test_dir, f"{asset}.npy")
            np.save(dst, np.load(src))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Directory containing L00*.las files"
    )
    parser.add_argument(
        "--output_root", required=True, help="Output root (data/toronto3d)"
    )
    args = parser.parse_args()

    las_files = sorted(glob.glob(os.path.join(args.dataset_root, "*.las")))
    if not las_files:
        raise FileNotFoundError(f"No .las files found under {args.dataset_root}")
    print(f"Found {len(las_files)} tiles: {[os.path.basename(f) for f in las_files]}")

    for las_path in las_files:
        process_tile(las_path, args.output_root)

    print(f"\nAll tiles written under {args.output_root}")


if __name__ == "__main__":
    main()
