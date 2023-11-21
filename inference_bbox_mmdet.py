from os.path import join, isfile, isdir
from os import listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser
import sys
import os
import numpy as np

import cv2
import torch

from mmdet.apis import DetInferencer
from tqdm import tqdm

model_name = 'rtmdet_tiny_8xb32-300e_coco'
checkpoint = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
device = 'cuda:0'

predictor = DetInferencer(model_name, checkpoint, device)

confidence_thres = 0.5

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
    outputs = predictor(l_stack)

    save_path = join(output_npz_dir, image_path.split('.')[0])

    pred_bbox = np.array(outputs["predictions"][0]["bboxes"])
    pred_scores = np.array(outputs["predictions"][0]["scores"])

    pred_bbox = pred_bbox[pred_scores > 0.5]
    pred_scores = pred_scores[pred_scores > 0.5]

    if args.filter_no_obj is True and pred_bbox.shape[0] == 0:
        print('delete {0}'.format(image_path))
        os.remove(join(input_dir, image_path))
        continue
    np.savez(save_path, bbox = pred_bbox, scores = pred_scores)