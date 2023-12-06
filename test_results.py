import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from fusion_dataset import *
from util import util
import os
from skimage import io
from skimage import img_as_ubyte


if __name__ == '__main__':
    '''
    Similar to train, but instead gets the metric results on LPIPS, SSIM, and PSNR.
    
    '''
    opt = TrainOptions().parse()
    if opt.stage == 'full':
        dataset = Training_Full_Dataset(opt)
    elif opt.stage == 'instance':
        dataset = Training_Instance_Dataset(opt)
    elif opt.stage == 'fusion':
        dataset = Training_Fusion_Dataset(opt)
    else:
        print('Error! Wrong stage selection!')
        exit()
        
    def collate_fn(batch):
        batch = batch[0]

        batch['cropped_rgb'] = [batch['cropped_rgb']]
        batch['full_gray'] = [batch['full_gray']]
        batch['full_rgb'] = [batch['full_rgb']]
        batch['cropped_gray'] = [batch['cropped_gray']]
        batch['box_info'] = [batch['box_info']]
        batch['box_info_2x'] = [batch['box_info_2x']]
        batch['box_info_4x'] = [batch['box_info_4x']]
        batch['box_info_8x'] = [batch['box_info_8x']]
        
        return batch
    
    # if using Training_Fusion_Dataset, use collate_fn=collate_fn
    if opt.stage == 'fusion':
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    else:
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    dataset_size = len(dataset)
    print('#testing images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)

    # opt.display_port = 8097
    # visualizer = Visualizer(opt)
    total_steps = 0

    lpips_vgg = lpips.LPIPS(net='vgg').cuda()

    lpips_val = 0
    psnr_val = 0
    ssim_val = 0

    total_steps = 0

    if opt.stage == 'full' or opt.stage == 'instance':
        for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
            total_steps += 1

            data_raw['rgb_img'] = [data_raw['rgb_img']]
            data_raw['gray_img'] = [data_raw['gray_img']]

            input_data = util.get_colorization_data(data_raw['gray_img'], opt, p=1.0, ab_thresh=0)
            gt_data = util.get_colorization_data(data_raw['rgb_img'], opt, p=1.0, ab_thresh=0)
            if gt_data is None:
                print("Issue 1")
                continue
            # print(gt_data['B'].shape, opt.batch_size)
            # if(gt_data['B'].shape[0] < opt.batch_size):
            #     print("Issue 2")
            #     continue
            input_data['B'] = gt_data['B']
            input_data['hint_B'] = gt_data['hint_B']
            input_data['mask_B'] = gt_data['mask_B']

            # visualizer.reset()
            model.set_input(input_data)
            model.forward()

            output = model.get_current_visuals()

            real = output['real'].detach()
            # fake = output['fake_reg'].detach()


            # lpips_val += lpips_vgg(real, fake).item()

            real = real.cpu().numpy()
            # fake = fake.cpu().numpy()

            # psnr_val += peak_signal_noise_ratio(real[0], fake[0], data_range=1.0)

            # ssim_val += structural_similarity(real[0], fake[0], channel_axis=0, data_range=1.0)

            # save the image in the results folder
            save_img_path = os.path.join(opt.train_img_dir, "base")
            # print(save_img_path)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            # save numpy array as RGB image
            save_img = np.transpose(torch.clamp(output['real'], 0.0, 1.0).cpu().data.numpy()[0], (1, 2, 0))
            io.imsave(os.path.join(save_img_path, data_raw['file_id'][0]), img_as_ubyte(save_img))
            

        print("LPIPS: ", lpips_val / total_steps)
        print("PSNR: ", psnr_val / total_steps)
        print("SSIM: ", ssim_val / total_steps)
    elif opt.stage == 'fusion':
        for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
            total_steps += 1
            box_info = data_raw['box_info'][0]
            box_info_2x = data_raw['box_info_2x'][0]
            box_info_4x = data_raw['box_info_4x'][0]
            box_info_8x = data_raw['box_info_8x'][0]
            cropped_input_data = util.get_colorization_data(data_raw['cropped_gray'], opt, p=1.0, ab_thresh=0)
            cropped_gt_data = util.get_colorization_data(data_raw['cropped_rgb'], opt, p=1.0, ab_thresh=10.0)
            full_input_data = util.get_colorization_data(data_raw['full_gray'], opt, p=1.0, ab_thresh=0)
            full_gt_data = util.get_colorization_data(data_raw['full_rgb'], opt, p=1.0, ab_thresh=10.0)
            if cropped_gt_data is None or full_gt_data is None:
                print("Issue?")
                continue
            cropped_input_data['B'] = cropped_gt_data['B']
            full_input_data['B'] = full_gt_data['B']
            model.set_input(cropped_input_data)
            model.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
            model.forward()

            output = model.get_current_visuals()

            real = output['real'].detach()
            fake = output['fake_reg'].detach()

            v = lpips_vgg(real, fake).item()

            # if v is nan, skip
            if torch.isnan(torch.tensor(v)):
                total_steps -= 1
                continue

            lpips_val += v

            real = real.cpu().numpy()
            fake = fake.cpu().numpy()

            v2 = peak_signal_noise_ratio(real[0], fake[0], data_range=1.0)

            # if v2 is nan, skip
            if torch.isnan(torch.tensor(v2)):
                total_steps -= 1
                lpips_val -= v
                continue

            psnr_val += v2

            ssim_val += structural_similarity(real[0], fake[0], channel_axis=0, data_range=1.0)

            # save the image in the results folder
            save_img_path = os.path.join(opt.train_img_dir, "results_fusion")
            # print(save_img_path)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            # save numpy array as RGB image
            save_img = np.transpose(torch.clamp(output['fake_reg'], 0.0, 1.0).cpu().data.numpy()[0], (1, 2, 0))
            io.imsave(os.path.join(save_img_path, data_raw['file_id']), img_as_ubyte(save_img))

            
            if total_steps % 100 == 0:
                print("LPIPS: ", lpips_val / total_steps)
                print("PSNR: ", psnr_val / total_steps)
                print("SSIM: ", ssim_val / total_steps)
        print("LPIPS: ", lpips_val / total_steps)
        print("PSNR: ", psnr_val / total_steps)
        print("SSIM: ", ssim_val / total_steps)
    else:
        print('Error! Wrong stage selection!')
        exit()
