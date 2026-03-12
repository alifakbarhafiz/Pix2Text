import os
import random
from pathlib import Path

import numpy as np
import torch

# paths (Colab: override these in notebook if needed)
DRIVE_MODELNET_PATH = "/content/ModelNet40"
EXTRACTED_DATASET_PATH = "/content/ModelNet40"
DRIVE_ZIP_PATH = "/content/drive/MyDrive/ModelNet40.zip"
PRECOMPUTED_PATH = Path("/content/point2text_features.npz")

USE_NUM_POINTS = 1024
BATCH_SIZE = 128
EPOCHS = 3
EMBED_DIM = 256
LR = 1e-3
NUM_WORKERS = 2
SAVE_DIR = "/content/point2text_checkpoints"
RANDOM_SEED = 42
NUM_EPOCHS = 200

device = None  # set by setup()


def setup(seed=None):
    """Makes SAVE_DIR, picks device, fixes RNG."""
    global device
    if seed is None:
        seed = RANDOM_SEED
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return device
