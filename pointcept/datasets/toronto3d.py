"""
Toronto3D Dataset

Large-scale outdoor MLS point cloud dataset (4 tiles L001..L004).
Preprocessed into per-scene `.npy` folders by
`preprocessing/toronto3d/preprocess_toronto3d.py`.

8 semantic classes (label 0 "unclassified" is remapped to ignore_index during
preprocessing). Standard benchmark split: train = L001/L003/L004, val/test = L002.

Author: Benchmarking setup for Pointcept
"""

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Toronto3DDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "strength",
        "segment",
    ]
    class2id = np.arange(8)
