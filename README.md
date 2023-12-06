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

## Instance Prediction
Please follow the command below to predict all the bounding boxes fo the images in `example` folder.
```
python inference_bbox.py --test_img_dir example
```
All the prediction results would save in `example_bbox` folder.

## Colorize Images
Please follow the command below to colorize all the images in `example` foler.
```
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
```
All the colorized results would save in `results` folder.

* Note: all the images would convert into L channel to colorize in [test_fusion.py's L51](test_fusion.py#L51)

## Training the Model
Please follow this [tutorial](README_TRAIN.md) to train the colorization model.

## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

## Acknowledgments
Our code borrows from the [InstColorization](https://github.com/ericsujw/InstColorization/tree/master) repository.
