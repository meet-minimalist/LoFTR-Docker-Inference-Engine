##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-23 11:44:59 pm
# @copyright MIT License
#
import os
import cv2
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
from typing import List, Tuple
import onnxruntime as ort
from kornia_moons.viz import draw_LAF_matches

from misc import load_img, preprocess


class InferenceEngine:
    """Wrapper class for LoFTR inference."""

    def __init__(self, model_path):
        """Initializer class to initiate the model."""
        self.sess = ort.InferenceSession(model_path)

    def __call__(
        self, img_path_1: str, img_path_2: str, op_dir: str
    ) -> Tuple[List, str]:
        """Function to make inference based on given image paths.

        Args:
            img_path_1 (str): Image path 1.
            img_path_2 (str): Image path 2.
            op_dir (str): Directory where the result will be stored.

        Returns:
            Tuple[List, str]: Tuple of model outputs and output path of the
                result image.
        """
        img_1 = load_img(img_path_1)
        img_2 = load_img(img_path_2)

        img_1_resized, img_1_gray = preprocess(img_1, (600, 375))
        img_2_resized, img_2_gray = preprocess(img_2, (600, 375))

        data = {"image_1": img_1_gray.numpy(), "image_2": img_2_gray.numpy()}
        result = self.sess.run(None, data)
        # keypoints0 : [N, 2]
        # keypoints1 : [N, 2]
        # confidence : [N]
        # batch_indexes: [N]

        mkpts0 = result[0]
        mkpts1 = result[1]

        Fm, inliers = cv2.findFundamentalMat(
            mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
        )
        inliers = inliers > 0

        fig, ax = draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.Tensor(mkpts0).view(1, -1, 2),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.Tensor(mkpts1).view(1, -1, 2),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1),
            ),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.utils.tensor_to_image(img_1_resized),
            K.utils.tensor_to_image(img_2_resized),
            inliers,
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": None,
                "feature_color": (0.2, 0.5, 1),
                "vertical": False,
            },
            return_fig_ax=True,
        )

        plt.axis("off")

        get_fname = lambda x: os.path.splitext(os.path.basename(x))[0]
        processed_img_name = (
            get_fname(img_path_1) + "_" + get_fname(img_path_2) + "_result.jpg"
        )
        processed_img_path = os.path.join(op_dir, processed_img_name)
        fig.savefig(processed_img_path, bbox_inches="tight", pad_inches=0)

        return result, processed_img_path
