import torch
import re
import logging
import os
import torch.nn.functional as F

from src.utils.io import *
from typing import Any, Callable, Dict, List


def checkCheckpointSaveName_exists(save_name: str):
    if os.path.exists(save_name):
        return True
    else:
        return False


def save_checkpoint(trainable_parameters: Dict, save_name: str) -> str:
    """
    Handle saving model.

    Args:
        trainable_parameters:
        save_name:

    Returns:
        save_name:
    """
    torch.save(
        trainable_parameters,
        save_name,
    )

    return save_name


def load_checkpoint(save_name) -> Dict:
    """
    Args:
        save_name:

    Returns:
        checkpoint:
    """
    checkpoint = torch.load(save_name)
    return checkpoint
