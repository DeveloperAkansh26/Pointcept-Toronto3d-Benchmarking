_base_ = ["../_base_/default_runtime.py"]

# logging: use tensorboard only (no wandb login required)
enable_wandb = False

# misc custom setting
batch_size = 6  # bs: total bs in all gpus (single L40S 46GB)
num_worker = 28  # 32-core box; SpUNet GPU step is ~0.25s so keep the dataloader fed
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
# SparseUNet (SpConv) baseline. Same Toronto3D data pipeline as the PTv3 config;
# only the backbone differs, so this is a controlled sparse-CNN vs transformer
# comparison. in_channels/num_classes live INSIDE the backbone for SpUNet.
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=7,  # coord(3) + color(3) + strength(1)
        num_classes=8,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
# Same AdamW + OneCycleLR schedule and epoch/eval_epoch loop as the PTv3 config,
# so the ONLY variable between the two benchmarks is the architecture.
# Pointcept semantics: `epoch` = total dataset passes, `eval_epoch` = number of
# outer epochs, per-epoch data loop = epoch // eval_epoch (= 100 here).
epoch = 10000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.006,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "Toronto3DDataset"
data_root = "data/toronto3d"

# grid size for outdoor MLS data (source is ~6cm subsampled)
grid_size = 0.05

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        "road",
        "road_marking",
        "natural",
        "building",
        "utility_line",
        "pole",
        "car",
        "fence",
    ],
    train=dict(
        type=dataset_type,
        split="train",  # L001, L003, L004
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # NOTE: ElasticDistortion intentionally omitted. It builds a dense 3D noise
            # grid over the cloud's spatial extent; fine for small S3DIS rooms but on a
            # hundreds-of-meters Toronto3D tile the grid is ~GBs/worker -> OOM. Matches
            # the PTv3 config (which also has no ElasticDistortion) for a fair comparison.
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "strength"),
            ),
        ],
        test_mode=False,
        # NOTE: loop is overwritten by the engine to epoch // eval_epoch (= 100 here)
    ),
    val=dict(
        type=dataset_type,
        split="val",  # L002
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color", "strength"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",  # L002 (mirror of val)
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1, 1]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
