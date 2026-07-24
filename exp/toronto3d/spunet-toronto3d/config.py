weight = 'exp/toronto3d/spunet-toronto3d/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 54250325
save_path = 'exp/toronto3d/spunet-toronto3d'
num_worker = 28
batch_size = 6
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 10000
eval_epoch = 100
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = None
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentor',
    backbone=dict(
        type='SpUNet-v1m1',
        in_channels=7,
        num_classes=8,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2)),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.006,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'Toronto3DDataset'
data_root = 'data/toronto3d'
grid_size = 0.05
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        'road', 'road_marking', 'natural', 'building', 'utility_line', 'pole',
        'car', 'fence'
    ],
    train=dict(
        type='Toronto3DDataset',
        split='train',
        data_root='data/toronto3d',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomDropout',
                dropout_ratio=0.2,
                dropout_application_ratio=0.2),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='ChromaticAutoContrast', p=0.2, blend_factor=None),
            dict(type='ChromaticTranslation', p=0.95, ratio=0.05),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='SphereCrop', point_max=120000, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'color', 'strength'))
        ],
        test_mode=False,
        loop=100),
    val=dict(
        type='Toronto3DDataset',
        split='val',
        data_root='data/toronto3d',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=('coord', 'color', 'strength'))
        ],
        test_mode=False),
    test=dict(
        type='Toronto3DDataset',
        split='test',
        data_root='data/toronto3d',
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'color', 'strength'))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1, 1]
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.9, 0.9]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1, 1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.1, 1.1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }]])))
