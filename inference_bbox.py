from os.path import join, isfile, isdir
from os import listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser

import numpy as np
import cv2

import torch
from tqdm import tqdm

from mmdet.apis import DetInferencer


# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)

# Choose to use a config
model_name = 'rtmdet_x_8xb32-300e_coco'
# Setup a checkpoint file to load
checkpoint = './checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'

# Set the device to be used for evaluation
device = 'cuda:0'

thres = 0.7

# Initialize the DetInferencer
predictor = DetInferencer(model_name, checkpoint, device)

parser = ArgumentParser()
parser.add_argument("--test_img_dir", type=str, default='example', help='testing images folder')
parser.add_argument('--filter_no_obj', action='store_true')
args = parser.parse_args()

input_dir = args.test_img_dir
image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
output_npz_dir = "{0}_bbox".format(input_dir)
if os.path.isdir(output_npz_dir) is False:
    print('Create path: {0}'.format(output_npz_dir))
    os.makedirs(output_npz_dir)

for image_path in tqdm(image_list):
    img = cv2.imread(join(input_dir, image_path))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
    outputs = predictor(l_stack, pred_score_thr=0.7)

    save_path = join(output_npz_dir, image_path.split('.')[0])
    pred_bbox = np.array(outputs["predictions"][0]["bboxes"])
    pred_scores = np.array(outputs["predictions"][0]["scores"])

    pred_bbox = pred_bbox[pred_scores >= thres]
    pred_scores = pred_scores[pred_scores >= thres]

    if args.filter_no_obj is True and pred_bbox.shape[0] == 0:
        print('delete {0}'.format(image_path))
        os.remove(join(input_dir, image_path))
        continue

    np.savez(save_path, bbox = pred_bbox, scores = pred_scores)