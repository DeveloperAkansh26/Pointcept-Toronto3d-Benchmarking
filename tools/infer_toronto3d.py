"""
Inference + `.las` export for the PTv3 Toronto3D model.

Runs a trained Pointcept PTv3 model on a Toronto3D `.las` tile and writes the
per-point predicted class back into a new `.las` that opens directly in
CloudCompare (color by the `pred` scalar field, or use `--recolor` for a
per-class RGB colormap).

Pointcept itself only saves raw `.npy` label arrays (no LAS/PLY export); this
script fills that gap. It mirrors the val pipeline in `pointcept/engines/test.py`
(single whole-cloud forward with GridSample + inverse mapping to full resolution).

Examples
--------
# Fast path: convert the already-saved TTA predictions for L002 to LAS (no GPU):
python tools/infer_toronto3d.py \
  --input  ~/Benchmarking/data/L002_6cm.las \
  --output ~/Benchmarking/data/L002_pred.las \
  --pred_npy exp/toronto3d/ptv3-toronto3d/result/L002_pred.npy --recolor

# Fresh inference on any (labeled or unlabeled) tile:
python tools/infer_toronto3d.py \
  --config configs/toronto3d/semseg-pt-v3m1-0-base.py \
  --weight exp/toronto3d/ptv3-toronto3d/model/model_best.pth \
  --input  /path/to/tile.las --output /path/to/tile_pred.las --recolor

Author: Benchmarking setup for Pointcept
"""

import argparse
import os
import sys

# Make `pointcept` importable when running this script directly from tools/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import laspy
except ImportError as e:
    raise ImportError("laspy is required: pip install 'laspy[lazrs]'") from e


# Training class id (0..7) -> (Toronto3D raw label, name, RGB color for --recolor).
# Matches preprocess_toronto3d.py: raw label = class + 1; 0 = Unclassified (ignore).
CLASSES = [
    ("road",         ( 90,  90,  90)),  # 0 -> raw 1
    ("road_marking", (255, 255,   0)),  # 1 -> raw 2
    ("natural",      ( 34, 139,  34)),  # 2 -> raw 3
    ("building",     (178,  34,  34)),  # 3 -> raw 4
    ("utility_line", (255, 140,   0)),  # 4 -> raw 5
    ("pole",         (138,  43, 226)),  # 5 -> raw 6
    ("car",          ( 30, 144, 255)),  # 6 -> raw 7
    ("fence",        (139,  69,  19)),  # 7 -> raw 8
]
NUM_CLASSES = len(CLASSES)


def get_field(las, *names):
    """Return the first available point dimension among `names`, else None."""
    available = set(las.point_format.dimension_names)
    for n in names:
        if n in available:
            return np.asarray(las[n])
    return None


def load_features(las):
    """Extract coord/color/strength EXACTLY like preprocess_toronto3d.py."""
    n = len(las.points)

    ex, ey, ez = get_field(las, "exact_x"), get_field(las, "exact_y"), get_field(las, "exact_z")
    if ex is not None and ey is not None and ez is not None:
        coord = np.stack([ex, ey, ez], axis=1).astype(np.float64)
    else:
        coord = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    coord = (coord - coord.min(axis=0, keepdims=True)).astype(np.float32)

    r, g, b = get_field(las, "red"), get_field(las, "green"), get_field(las, "blue")
    if r is not None:
        color = np.stack([r, g, b], axis=1).astype(np.float32)
        if color.max() > 255:
            color = color / 256.0
    else:
        color = np.zeros((n, 3), dtype=np.float32)

    intensity = get_field(las, "intensity")
    if intensity is not None:
        intensity = intensity.astype(np.float32)
        imax = intensity.max()
        strength = (intensity / imax) if imax > 0 else intensity
        strength = strength.reshape(-1, 1).astype(np.float32)
    else:
        strength = np.zeros((n, 1), dtype=np.float32)

    return coord, color, strength


def run_model(cfg_path, weight_path, coord, color, strength, grid_size):
    """Faithful single whole-cloud forward, returns full-resolution class ids (0..7)."""
    import torch
    import torch.nn.functional as F
    from pointcept.models import build_model
    from pointcept.datasets.transform import Compose
    from pointcept.utils.config import Config

    cfg = Config.fromfile(cfg_path)

    # Val pipeline (must match configs/toronto3d/...base.py `data.val.transform`):
    # CenterShift brackets the GridSample because `coord` is an input feature, so
    # the model must see centered coordinates exactly as it did in training.
    transform = Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "inverse"),
             feat_keys=("coord", "color", "strength")),
    ])

    data_dict = transform(dict(coord=coord.copy(), color=color.copy(), strength=strength.copy()))
    inverse = data_dict.pop("inverse")  # voxel -> full-resolution index
    if hasattr(inverse, "cpu"):
        inverse = inverse.cpu().numpy()

    model = build_model(cfg.model).cuda().eval()
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys when loading weights (e.g. {missing[:3]})")

    input_dict = {k: (v.cuda(non_blocking=True) if hasattr(v, "cuda") else v)
                  for k, v in data_dict.items()}
    with torch.no_grad():
        logits = model(input_dict)["seg_logits"]        # (n_voxel, num_classes)
        voxel_pred = F.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()

    return voxel_pred[inverse]                            # (N,) full resolution


def write_las(in_path, out_path, pred_class, recolor):
    """Copy the input LAS and attach predictions as a `pred` scalar dim (+optional RGB)."""
    las = laspy.read(in_path)
    n = len(las.points)
    assert len(pred_class) == n, f"pred ({len(pred_class)}) != points ({n})"

    raw_label = (pred_class.astype(np.int32) + 1).astype(np.uint8)  # 0..7 -> Toronto3D 1..8

    if "pred" not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.uint8,
                                                 description="PTv3 predicted class 1..8"))
    las.pred = raw_label

    if recolor:
        colors = np.array([c for _, c in CLASSES], dtype=np.uint16) * 256  # 8-bit -> 16-bit
        rgb = colors[pred_class.astype(np.int64)]
        las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    las.write(out_path)


def report_accuracy(las_in, pred_class):
    """If the input LAS carries ground-truth `label`, print mIoU / accuracy."""
    las = laspy.read(las_in)
    label = get_field(las, "label", "Label", "scalar_Label", "classification")
    if label is None:
        return
    gt = np.where(label.astype(np.int64) >= 1, label.astype(np.int64) - 1, -1)  # -> 0..7 / -1
    valid = gt >= 0
    if valid.sum() == 0:
        return
    ious, accs = [], []
    for c in range(NUM_CLASSES):
        p, g = pred_class == c, gt == c
        inter = np.logical_and(p, g & valid).sum()
        union = np.logical_and(np.logical_or(p, g), valid).sum()
        ious.append(inter / union if union else float("nan"))
        accs.append(inter / g.sum() if g.sum() else float("nan"))
    overall = (pred_class[valid] == gt[valid]).mean()
    print("\n=== Accuracy vs ground-truth label ===")
    for c, (name, _) in enumerate(CLASSES):
        print(f"  {name:<13} IoU {ious[c]:.4f}  Acc {accs[c]:.4f}")
    print(f"  {'mIoU':<13} {np.nanmean(ious):.4f}   allAcc {overall:.4f}")


def main():
    ap = argparse.ArgumentParser(description="PTv3 Toronto3D inference -> .las export")
    ap.add_argument("--input", required=True, help="input .las tile")
    ap.add_argument("--output", required=True, help="output .las with predictions")
    ap.add_argument("--config", default="configs/toronto3d/semseg-pt-v3m1-0-base.py")
    ap.add_argument("--weight", default="exp/toronto3d/ptv3-toronto3d/model/model_best.pth")
    ap.add_argument("--pred_npy", default=None,
                    help="use a precomputed prediction .npy (0..7, per point) instead of the model")
    ap.add_argument("--grid_size", type=float, default=0.05)
    ap.add_argument("--recolor", action="store_true", help="also write per-class RGB colormap")
    args = ap.parse_args()

    las = laspy.read(args.input)
    n = len(las.points)
    print(f"[read] {args.input}: {n} points")

    if args.pred_npy:
        pred_class = np.load(args.pred_npy).astype(np.int64)
        assert len(pred_class) == n, f"pred_npy ({len(pred_class)}) != points ({n})"
        print(f"[pred] loaded precomputed predictions from {args.pred_npy}")
    else:
        coord, color, strength = load_features(las)
        print(f"[model] running PTv3 forward (grid_size={args.grid_size}) ...")
        pred_class = run_model(args.config, args.weight, coord, color, strength, args.grid_size)

    uniq, cnts = np.unique(pred_class, return_counts=True)
    dist = {CLASSES[int(u)][0]: int(c) for u, c in zip(uniq, cnts) if 0 <= u < NUM_CLASSES}
    print(f"[pred] class distribution: {dist}")

    report_accuracy(args.input, pred_class)

    write_las(args.input, args.output, pred_class, args.recolor)
    print(f"\n[done] wrote {args.output}")
    print("       View in CloudCompare: color by scalar field 'pred' (1..8)"
          + (" or use the RGB colormap (--recolor applied)." if args.recolor else "."))


if __name__ == "__main__":
    main()
