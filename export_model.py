##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-25 7:09:07 pm
# @copyright MIT License
#

import os
import torch
import onnx
from onnxsim import simplify
import onnxruntime as ort
from misc import load_img, preprocess
import kornia.feature as KF

os.makedirs("./model", exist_ok=True)
model_output_path = "./model/loftr.onnx"
optimized_model_path = "./model/loftr_opt.onnx"

img_1 = load_img("./temp/kn_church-2.jpg")
img_2 = load_img("./temp/kn_church-8.jpg")

img_1_resized, img_1_gray = preprocess(img_1, (600, 375))
img_2_resized, img_2_gray = preprocess(img_2, (600, 375))

matcher = KF.LoFTR(pretrained="outdoor")

op = matcher(img_1_gray, img_2_gray)

torch.onnx.export(matcher, (img_1_gray, img_2_gray), model_output_path, input_names=["image_1", "image_2"], output_names=["kpt0", "kpt1", "conf", "batch_idx"])

onnx_model = onnx.load(model_output_path)

onnx.checker.check_model(onnx_model)
updated_model, status = simplify(onnx_model)
onnx.save(updated_model, model_output_path)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_options.optimized_model_filepath = optimized_model_path
sess = ort.InferenceSession(model_output_path, sess_options)
