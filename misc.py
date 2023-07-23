##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-23 6:36:19 pm
# @copyright MIT License
#

import torch
import kornia as K
from typing import Tuple


def load_img(img_path: str) -> torch.Tensor:
    """Function to load image as torch tensor from given image path.

    Args:
        img_path (str): Image path.

    Returns:
        torch.Tensor: Torch tensor instance.
    """
    # Returned image will be 4d array of shape [1, 3, H, W]
    img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32)[None, ...]
    return img


def preprocess(img: torch.Tensor, tar_hw: Tuple) -> Tuple[torch.Tensor]:
    """Function to preprocess image for LoFTR inference.

    Args:
        img (torch.Tensor): Image loaded as a torch tensor. shape: [1, 3, H, W]

    Returns:
        torch.Tensor: Tuple of resized RGB image and resized gray image.
    """
    img_res = K.geometry.resize(img, tar_hw, antialias=True)
    img_gray = K.color.rgb_to_grayscale(img_res)
    return img_res, img_gray
