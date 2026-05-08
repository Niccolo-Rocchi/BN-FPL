import random
import shutil
import sys
import numpy as np
from pathlib import Path

import pyagrum as gum

IN_PYTEST = "pytest" in sys.modules


# Set global seed
def set_seed():

    np.random.seed(42)
    random.seed(42)
    gum.initRandom(42)


# Create an empty directory
def create_clean_dir(path: Path):

    # If directory exists, clean it
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    # Else, create a new one
    else:
        path.mkdir(parents=True, exist_ok=True)


# Get output path
def get_cur_dir(config):

    root_path = get_root_path()
    cur_dir = config["cur_dir"]

    return root_path / cur_dir


# Get root (project) directory
def get_root_path():
    return Path(__file__).resolve().parents[1]


# Only perform an `assert` if code is running in `pytest`
def safe_assert(condition):
    if IN_PYTEST:
        assert condition
