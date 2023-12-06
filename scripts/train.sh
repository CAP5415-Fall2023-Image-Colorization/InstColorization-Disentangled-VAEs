DATASET_DIR=train_data/train2017

# Stage 1: Training Full Image Colorization
mkdir ./checkpoints/coco_full
python train.py --stage full --name coco_full --sample_p 1.0 --niter 15 --niter_decay 0 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size 8 --display_ncols 3 --display_freq 100 --print_freq 100 --train_img_dir train_data/train2017
# Stage 2: Training Instance Image Colorization

mkdir ./checkpoints/coco_instance
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_instance/
python train.py --stage instance --name coco_instance  --sample_p 1.0 --niter 15 --niter_decay 0 --load_model --lr 0.0005 --model train --fineSize 256 --batch_size 8 --display_ncols 3 --display_freq 100 --print_freq 100 --train_img_dir train_data/train2017

# Stage 3: Training Fusion Module
mkdir ./checkpoints/coco_mask
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GF.pth
cp ./checkpoints/coco_instance/latest_net_G.pth ./checkpoints/coco_mask/latest_net_G.pth
cp ./checkpoints/coco_full/latest_net_G.pth ./checkpoints/coco_mask/latest_net_GComp.pth
python train.py --stage fusion --name coco_mask --sample_p 1.0 --niter 5 --niter_decay 0 --lr 0.00001 --model train --load_model --display_ncols 4 --fineSize 256 --batch_size 8 --display_ncols 3 --display_freq 100 --print_freq 100 --train_img_dir train_data/train2017
