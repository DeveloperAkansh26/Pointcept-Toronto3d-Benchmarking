# Toronto3D Semantic Segmentation Benchmark

A reproducible benchmark of point-cloud semantic segmentation models on the
[**Toronto3D**](https://github.com/WeikaiTan/Toronto-3D) outdoor mobile-laser (MLS) dataset,
built on top of the [**Pointcept**](https://github.com/Pointcept/Pointcept) framework.

Two architecturally different models are trained and evaluated on the **exact same data, split, and
training recipe**, so the comparison isolates the architecture:

| Model | Type | L002 mIoU (TTA) | allAcc | mAcc | Weights |
|---|---|:--:|:--:|:--:|---|
| **PTv3** (Point Transformer V3) | Transformer | **0.8150** | 0.9709 | 0.8742 | `exp/toronto3d/ptv3-toronto3d/model/model_best.pth` |
| **SpUNet** (SparseUNet v1m1) | Sparse CNN | 0.7986 | 0.9708 | 0.8570 | `exp/toronto3d/spunet-toronto3d/model/model_best.pth` |

Both are evaluated with **test-time augmentation (TTA) voting** on tile **L002** (the standard
Toronto3D held-out tile), using the best-validation checkpoint.

### Per-class IoU (L002, TTA)

| | road | road_marking | natural | building | utility_line | pole | car | fence |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **PTv3** | 0.961 | **0.637** | 0.971 | 0.912 | **0.860** | 0.795 | **0.915** | **0.471** |
| **SpUNet** | **0.966** | 0.600 | 0.967 | **0.916** | 0.839 | **0.807** | 0.900 | 0.394 |

PTv3 wins overall by ~1.6 mIoU points, mostly on the hard sparse classes (**fence**, **road_marking**);
SpUNet is competitive on the common planar classes and trains substantially faster.

---

## What this is

This repository is a **fork of [Pointcept](https://github.com/Pointcept/Pointcept)** with everything
needed to run the Toronto3D benchmark added on top. The upstream Pointcept documentation (all its
models, datasets, and features) is preserved in **[POINTCEPT_README.md](POINTCEPT_README.md)**.

Files added for this benchmark:

| Path | Purpose |
|---|---|
| [`configs/toronto3d/semseg-pt-v3m1-0-base.py`](configs/toronto3d/semseg-pt-v3m1-0-base.py) | PTv3 config (7-ch input, 8 classes) |
| [`configs/toronto3d/semseg-spunet-v1m1-0-base.py`](configs/toronto3d/semseg-spunet-v1m1-0-base.py) | SpUNet config (same data/recipe as PTv3) |
| [`pointcept/datasets/toronto3d.py`](pointcept/datasets/toronto3d.py) | `Toronto3DDataset` class |
| [`pointcept/datasets/preprocessing/toronto3d/preprocess_toronto3d.py`](pointcept/datasets/preprocessing/toronto3d/preprocess_toronto3d.py) | LAS → per-scene `.npy` converter |
| [`tools/infer_toronto3d.py`](tools/infer_toronto3d.py) | Run a trained model on a `.las` and export a labeled `.las` |
| `exp/toronto3d/{ptv3,spunet}-toronto3d/` | Trained checkpoints, logs, configs, predictions |

**Data model:** each point has 7 input channels = `coord(3) + color(3) + strength(1)` and one of
**8 semantic classes**. Standard split: **train = L001/L003/L004**, **val = test = L002**.

---

## Setup

Requires an **NVIDIA GPU with compute capability ≥ 8.0** (Ampere/Ada — PTv3 uses flash-attention),
`conda`, `git`, and `git-lfs`.

### 1. Clone + pull the weights (Git LFS)

```bash
# install git-lfs if needed (no sudo required):
conda install -n base -c conda-forge git-lfs -y && git lfs install

git clone https://github.com/DeveloperAkansh26/Pointcept-Toronto3d-Benchmarking.git
cd Pointcept-Toronto3d-Benchmarking
git lfs pull                                  # fetches the two model_best.pth (~1 GB)
ls -lh exp/toronto3d/*/model/model_best.pth   # should be ~450–550 MB each
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml           # python 3.10, torch 2.5.x, cuda 12.4, gcc 13
conda activate pointcept-torch2.5.0-cu12.4
```

### 3. Install the extra dependencies (not in `environment.yml`)

```bash
pip install spconv-cu124                                                    # 2.3.8
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch_geometric
pip install "laspy[lazrs]"                                                  # 2.7.0  (LAS I/O)

# flash-attention (REQUIRED by PTv3; non-flash OOMs on large tiles) — prebuilt wheel, no compile:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# open3d runtime libs on headless boxes:
conda install -c conda-forge xorg-libx11 xorg-libxext xorg-libxrender xorg-libxfixes libgl mesalib -y
```

### 4. Build the `pointops` CUDA extension (required by PTv3)

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"   # set to YOUR GPU: A100=8.0, RTX30xx=8.6, L40S/RTX40xx=8.9
cd libs/pointops && python setup.py install && cd ../..
```

### 5. Sanity check

```bash
python -c "import torch, spconv, flash_attn, laspy, pointops, torch_scatter, torch_cluster; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
# expect: torch 2.5.x+cu124 | cuda True
```

> **SpUNet-only shortcut:** if you only want SpUNet, you can **skip flash-attn and the `pointops`
> build** — SpUNet needs only `spconv` + `laspy`. PTv3 requires both.

---

## Quick start — run inference on a custom LAS → labeled LAS

The one-shot tool builds the model from a config, loads the weights, runs a full-cloud forward, and
writes a new `.las` with a per-point `pred` class field (+ optional per-class RGB for CloudCompare).

**PTv3 (best model):**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tools/infer_toronto3d.py \
  --config configs/toronto3d/semseg-pt-v3m1-0-base.py \
  --weight exp/toronto3d/ptv3-toronto3d/model/model_best.pth \
  --input  /path/to/your_input.las \
  --output /path/to/your_output.las \
  --recolor
```

**SpUNet (faster, lighter):**
```bash
python tools/infer_toronto3d.py \
  --config configs/toronto3d/semseg-spunet-v1m1-0-base.py \
  --weight exp/toronto3d/spunet-toronto3d/model/model_best.pth \
  --input  /path/to/your_input.las \
  --output /path/to/your_output.las \
  --recolor
```

**Output:** a copy of your input `.las` with a `pred` scalar field per point (Toronto3D class **1–8**),
and — with `--recolor` — RGB overwritten by a per-class colormap. Open in **CloudCompare** and color
by RGB or by the scalar field `pred`. If your input LAS carries a ground-truth `label` field, the
script also prints **mIoU / per-class IoU**.

**Notes on the input LAS:**
- The models are **7-channel: XYZ + RGB color + intensity**. For best results your LAS should have
  color *and* intensity; missing channels are zero-filled (accuracy drops).
- Trained on Toronto3D (colored urban street MLS, ~6 cm spacing) — expect strong results on similar
  data; very different scenes degrade.
- This does a **single whole-cloud forward** (no TTA) — the ~0.796 quality point for PTv3, fine for
  prediction. Toronto3D-sized tiles (~2–6 M points) fit comfortably on ≥16 GB VRAM.

---

## Reproduce the benchmark from scratch

You need the raw Toronto3D tiles `L001_6cm.las … L004_6cm.las` (with XYZ, RGB, intensity, and the
`label` extra dimension).

### 1. Preprocess LAS → `.npy`

```bash
python pointcept/datasets/preprocessing/toronto3d/preprocess_toronto3d.py \
  --dataset_root /path/to/las_tiles \
  --output_root  data/toronto3d
# writes data/toronto3d/{train,val,test}/L00x/{coord,color,strength,segment}.npy
# split: L001/L003/L004 -> train, L002 -> val (mirrored to test)
```

### 2. Train

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PTv3
sh scripts/train.sh -p python -g 1 -d toronto3d -c semseg-pt-v3m1-0-base   -n ptv3-toronto3d
# SpUNet
sh scripts/train.sh -p python -g 1 -d toronto3d -c semseg-spunet-v1m1-0-base -n spunet-toronto3d
```

Training writes to `exp/toronto3d/<name>/`, evaluates on L002 each epoch, saves `model_best.pth` on
best val mIoU, and **auto-runs the TTA test** on `model_best` at the end. Launch detached
(`nohup … &`) and tail `exp/toronto3d/<name>/train.log`.

> **Loop semantics** (Pointcept): `epoch=10000, eval_epoch=100` means 100 outer epochs, each looping
> the 3-tile train set `epoch // eval_epoch = 100` times (~50 iters/epoch at batch size 6).

### 3. Evaluate (TTA voting on L002)

```bash
sh scripts/test.sh -p python -g 1 -d toronto3d -c semseg-pt-v3m1-0-base -n ptv3-toronto3d
```
Logs mIoU / allAcc / per-class IoU and writes `exp/toronto3d/<name>/result/L002_pred.npy`. The
training run already performs this once; re-run only to re-evaluate.

---

## Model weights

| File | Location | How to get it |
|---|---|---|
| `model_best.pth` (best-val, for inference) | in-repo under `exp/toronto3d/*/model/` | `git lfs pull` |
| `model_last.pth` (final epoch, for resuming training) | **GitHub Release `v1.0-toronto3d`** | download links below |

Release assets (free, not in the clone — download separately if you want to resume training):
- **PTv3:** https://github.com/DeveloperAkansh26/Pointcept-Toronto3d-Benchmarking/releases/download/v1.0-toronto3d/ptv3_model_last.pth
- **SpUNet:** https://github.com/DeveloperAkansh26/Pointcept-Toronto3d-Benchmarking/releases/download/v1.0-toronto3d/spunet_model_last.pth

---

## Classes & label mapping

`preprocess_toronto3d.py` maps the raw Toronto3D label (1–8; 0 = unclassified → ignored) to a
training id (0–7). `tools/infer_toronto3d.py --recolor` uses the RGB below.

| Train id | Raw label | Class | Recolor RGB |
|:--:|:--:|---|---|
| 0 | 1 | road | (90, 90, 90) |
| 1 | 2 | road_marking | (255, 255, 0) |
| 2 | 3 | natural | (34, 139, 34) |
| 3 | 4 | building | (178, 34, 34) |
| 4 | 5 | utility_line | (255, 140, 0) |
| 5 | 6 | pole | (138, 43, 226) |
| 6 | 7 | car | (30, 144, 255) |
| 7 | 8 | fence | (139, 69, 19) |

---

## Credits & license

- Framework: **[Pointcept](https://github.com/Pointcept/Pointcept)** (Apache-2.0) — see
  [POINTCEPT_README.md](POINTCEPT_README.md) and [LICENSE](LICENSE).
- Models: **Point Transformer V3** (Wu et al., CVPR 2024) and **SparseUNet / MinkUNet** as
  implemented in Pointcept.
- Dataset: **[Toronto3D](https://github.com/WeikaiTan/Toronto-3D)** (Tan et al., CVPRW 2020).

This benchmark's added code follows the upstream Apache-2.0 license.
