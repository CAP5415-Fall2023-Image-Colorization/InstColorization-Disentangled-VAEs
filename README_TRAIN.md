## Prerequisites
* [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
* Python3
* Pytorch >= 1.5
* MMDetection
* OpenCV-Python
* Pillow/scikit-image
* Please refer to the [env.yml](env.yml) for detail dependencies.

## Getting Started
1. Follow the direction for installing [MMDetection](https://github.com/open-mmlab/mmdetection).
   
    Link for installation instruction can be found at https://mmdetection.readthedocs.io/en/latest/get_started.html.

2. Clone this repo:
```sh
git clone https://github.com/CAP5415-Fall2023-Image-Colorization/InstColorization-Disentangled-VAEs
cd InstColorization-Disentangled-VAEs
```
1. Install [conda](https://www.anaconda.com/).
2. Install all the dependencies
```sh
conda env create --file env.yml
```
1. Switch to the conda environment
```sh
conda activate instacolorization
```
1. Install other dependencies
```sh
sh scripts/install.sh
```

## Dataset Preparation
### COCOStuff
1. Download and unzip the COCOStuff training set:
```sh
sh scripts/prepare_cocostuff.sh
```
2. Now the COCOStuff train set would place in [train_data](train_data).

### Your own Dataset
1. If you want to train on your dataset, you should change the dataset path in [scripts/prepare_train_box.sh's L1](scripts/prepare_train_box.sh#L1) and in [scripts/train.sh's L1](scripts/train.sh#L1).

## Instance Prediction
Please follow the command below to predict all the bounding boxes fo the images in `${DATASET_DIR}` folder.
```sh
sh scripts/prepare_train_box.sh
```
All the prediction results would save in `${DATASET_DIR}_bbox` folder.

## Training the Instance-aware Image Colorization model
Simply run the following command, then the training pipeline would get start.
```sh
sh scripts/train.sh
```
To view training results and loss plots, run `visdom -port 8097` and click the URL http://localhost:8097.

This is a 3 stage training process.
1. We would start to train our full image colorization branch with random initial weights.
2. We would use the full image colorization branch's weight as our instance colorization branch's pretrained weight.
3. Finally, we would train the fusion module with the full image and instance models.

## Testing the Instance-aware Image Colorization model
1. Our model's weight would place in [checkpoints/coco_mask](checkpoints/coco_mask).
2. Please follow the command below to colorize all the images in `example` foler based on the weight placed in `coco_mask`.

    ```
    python test_results.py --name coco_<mask, instance, full> --sample_p 1.0 --model train --stage <fusion, instance, full> --fineSize 256 --train_img_dir example --results_img_dir results
    ```
    All the colorized results would save in `results` folder.

## Acknowledgments
Our code borrows from the [InstColorization](https://github.com/ericsujw/InstColorization/tree/master) repository.
