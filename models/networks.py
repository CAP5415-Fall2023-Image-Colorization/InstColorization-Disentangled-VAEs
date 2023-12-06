import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler


def get_norm_layer(norm_type='instance'):
    '''
    Gets the normalization layer for the model.
    '''
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    '''
    Set up Learning rate scheduler. LambdaLR is used for this project with a linear decay.
    Other options include StepLR and ReduceLROnPlateau.
    '''
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='xavier', gain=0.02):
    '''
    Load model weights into the corresponding model architecture.
    '''
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', gpu_ids=[]):
    '''
    Load model to GPU.
    '''
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='xavier', gpu_ids=[], use_tanh=True, classification=False):
    '''
    Function for generating specific model types and initializing them.
    '''
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG =='siggraph':
        netG = SIGGRAPHGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=use_tanh, classification=classification)
    elif which_model_netG =='instance':
        netG = InstanceGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=use_tanh, classification=classification)
    elif which_model_netG == 'fusion':
        netG = FusionGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=use_tanh, classification=classification)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


class HuberLoss(nn.Module):
    '''
    Loss function for base image colorization. Smooth L1 loss aka Huber loss.
    '''
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1, ig1, ig2):
        # replace any nan values with 0
        # in0[torch.isnan(in0)] = 0
        # in1[torch.isnan(in1)] = 0

        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)

        val = torch.sum(loss,dim=1,keepdim=True)

        return val, val, val
    
class VAELoss(nn.Module):
    '''
    Loss function for VAE, includes reconstruction loss and KL divergence loss. Reconstruction loss is the smooth L1 loss aka Huber loss.
    '''
    def __init__(self, delta=.01, kl_weight=0.00025):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.delta = delta

    def __call__(self, reconstruction, input, mu, logvar):

        mask = torch.zeros_like(reconstruction)
        mann = torch.abs(reconstruction-input)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)

        reconstruction_loss = torch.sum(loss,dim=1,keepdim=True)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

        return (reconstruction_loss + self.kl_weight * kld_loss, reconstruction_loss, kld_loss)


class SIGGRAPHGenerator(nn.Module):
    '''
    Architecture for full image colorization.

    '''
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(SIGGRAPHGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Encoder
        self.model1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(64),

        )

        self.model2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(128),
        )

        self.model3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(256),

        )

        self.model4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(512),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Mu and Var
        self.model5_mu = nn.Linear(512 * 32 * 32, 64)
        self.model5_logvar = nn.Linear(512 * 32 * 32, 64)

        # Decoder
        self.decoder_input = nn.Linear(64, 512 * 32 * 32)

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(512),
        )

        # Here we are both deconvolving AND using a U-Net style skip connection from layer 3
        self.model7_upscale = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),) 
        self.model7_skip_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.model7 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(256),
        )

        self.model8_upscale = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),)
        self.model8_skip_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.model8 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(128),
        )

        self.model9_upscale = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),)
        self.model9_skip_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.model9 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(negative_slope=.2),
        )

        self.model_out = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),nn.Tanh())
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def forward(self, input_A, input_B, mask_B):
        init_image = torch.cat((input_A, input_B, mask_B), dim=1)

        conv1_2 = self.model1(init_image)
        conv2_3 = self.model2(conv1_2)
        conv3_4 = self.model3(conv2_3)
        conv4_5 = self.model4(conv3_4)

        mu = self.model5_mu(conv4_5.view(-1, 512 * 32 * 32))
        logvar = self.model5_logvar(conv4_5.view(-1, 512 * 32 * 32))

        # z = self.reparameterize(mu, logvar)

        # decoder_input = self.decoder_input(z)
        # decoder_input = decoder_input.view(-1, 512, 32, 32)

        conv6_7 = self.model6(conv4_5)
        # print(conv6_7.shape, conv3_4.shape)
        conv7_up = self.model7_upscale(conv6_7) + self.model7_skip_3(conv3_4)
        conv7_8 = self.model7(conv7_up)
        conv8_up = self.model8_upscale(conv7_8) + self.model8_skip_2(conv2_3)
        conv8_9 = self.model8(conv8_up)
        conv9_up = self.model9_upscale(conv8_9) + self.model9_skip_1(conv1_2)
        conv9_10 = self.model9(conv9_up)
        out_reg = self.model_out(conv9_10)

        return (out_reg, mu, logvar)


        # conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        # conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        # conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        # conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        # conv5_3 = self.model5(conv4_3)
        # conv6_3 = self.model6(conv5_3)
        # conv7_3 = self.model7(conv6_3)
        # conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        # conv8_3 = self.model8(conv8_up)

        # if(self.classification):
        #     out_class = self.model_class(conv8_3)
        #     conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
        #     conv9_3 = self.model9(conv9_up)
        #     conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
        #     conv10_2 = self.model10(conv10_up)
        #     out_reg = self.model_out(conv10_2)
        # else:
        #     out_class = self.model_class(conv8_3.detach())

        #     conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        #     conv9_3 = self.model9(conv9_up)
        #     conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        #     conv10_2 = self.model10(conv10_up)
        #     out_reg = self.model_out(conv10_2)

        # return (out_class, out_reg)


class FusionGenerator(nn.Module):
    '''
    Class for fused full image and instance model image colorization.
    '''
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(FusionGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

                # Encoder
        self.model1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(64),

        )

        self.weight_layer = WeightGenerator(64)


        self.model2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(128),
        )

        self.weight_layer2 = WeightGenerator(128)

        self.model3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(256),

        )

        self.weight_layer3 = WeightGenerator(256)

        self.model4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(512),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.weight_layer4 = WeightGenerator(512)

        # Mu and Var
        self.model5_mu = nn.Linear(512 * 32 * 32, 64)
        self.model5_logvar = nn.Linear(512 * 32 * 32, 64)

        # Decoder
        self.decoder_input = nn.Linear(64, 512 * 32 * 32)

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(512),
        )

        self.weight_layer6 = WeightGenerator(512)

        # Here we are both deconvolving AND using a U-Net style skip connection from layer 3
        self.model7_upscale = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),) 
        self.model7_skip_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.weight_layer7_1 = WeightGenerator(256)

        self.model7 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(256),
        )

        self.weight_layer7_2 = WeightGenerator(256)

        self.model8_upscale = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),)
        self.model8_skip_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.weight_layer8_1 = WeightGenerator(128)

        self.model8 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(128),
        )

        self.weight_layer8_2 = WeightGenerator(128)

        self.model9_upscale = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),)
        self.model9_skip_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.weight_layer9_1 = WeightGenerator(64)

        self.model9 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(negative_slope=.2),
        )

        self.weight_layer9_2 = WeightGenerator(64)

        self.model_out = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),nn.Tanh())

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def forward(self, input_A, input_B, mask_B, instance_feature, box_info_list):

        init_image = torch.cat((input_A, input_B, mask_B), dim=1)

        conv1_2 = self.model1(init_image)
        conv1_2 = self.weight_layer(instance_feature['conv1_2'], conv1_2, box_info_list[0])

        conv2_3 = self.model2(conv1_2)
        conv2_3 = self.weight_layer2(instance_feature['conv2_3'], conv2_3, box_info_list[1])

        conv3_4 = self.model3(conv2_3)
        conv3_4 = self.weight_layer3(instance_feature['conv3_4'], conv3_4, box_info_list[2])

        conv4_5 = self.model4(conv3_4)
        conv4_5 = self.weight_layer4(instance_feature['conv4_5'], conv4_5, box_info_list[3])

        mu = self.model5_mu(conv4_5.view(-1, 512 * 32 * 32))
        logvar = self.model5_logvar(conv4_5.view(-1, 512 * 32 * 32))

        z = self.reparameterize(mu, logvar)

        decoder_input = self.decoder_input(z)
        decoder_input += instance_feature['decoder_input']

        decoder_input = decoder_input.view(-1, 512, 32, 32)

        conv6_7 = self.model6(decoder_input)
        conv6_7 = self.weight_layer6(instance_feature['conv6_7'], conv6_7, box_info_list[3])

        conv7_up = self.model7_upscale(conv6_7) + self.model7_skip_3(conv3_4)
        conv7_up = self.weight_layer7_1(instance_feature['conv7_up'], conv7_up, box_info_list[2])

        conv7_8 = self.model7(conv7_up)
        conv7_8 = self.weight_layer7_2(instance_feature['conv7_8'], conv7_8, box_info_list[2])

        conv8_up = self.model8_upscale(conv7_8) + self.model8_skip_2(conv2_3)
        conv8_up = self.weight_layer8_1(instance_feature['conv8_up'], conv8_up, box_info_list[1])

        conv8_9 = self.model8(conv8_up)
        conv8_9 = self.weight_layer8_2(instance_feature['conv8_9'], conv8_9, box_info_list[1])

        conv9_up = self.model9_upscale(conv8_9) + self.model9_skip_1(conv1_2)
        conv9_up = self.weight_layer9_1(instance_feature['conv9_up'], conv9_up, box_info_list[0])

        conv9_10 = self.model9(conv9_up)
        conv9_10 = self.weight_layer9_2(instance_feature['conv9_10'], conv9_10, box_info_list[0])

        out_reg = self.model_out(conv9_10)

        return (out_reg, mu, logvar)


class WeightGenerator(nn.Module):
    '''
    Class for resizing and fusing instance layer parameters and full image parameters together with correct padding.
    '''
    def __init__(self, input_ch, inner_ch=16):
        super(WeightGenerator, self).__init__()
        self.simple_instance_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.simple_bg_conv = nn.Sequential(
            nn.Conv2d(input_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.normalize = nn.Softmax(1)
    
    def resize_and_pad(self, feauture_maps, info_array):
        feauture_maps = torch.nn.functional.interpolate(feauture_maps, size=(info_array[5], info_array[4]), mode='bilinear')
        feauture_maps = torch.nn.functional.pad(feauture_maps, (info_array[0], info_array[1], info_array[2], info_array[3]), "constant", 0)
        return feauture_maps
    
    def forward(self, instance_feature, bg_feature, box_info):
        mask_list = []
        featur_map_list = []
        mask_sum_for_pred = torch.zeros_like(bg_feature)[:1, :1]
        for i in range(instance_feature.shape[0]):
            tmp_crop = torch.unsqueeze(instance_feature[i], 0)
            conv_tmp_crop = self.simple_instance_conv(tmp_crop)
            pred_mask = self.resize_and_pad(conv_tmp_crop, box_info[i])
            
            tmp_crop = self.resize_and_pad(tmp_crop, box_info[i])

            mask = torch.zeros_like(bg_feature)[:1, :1]
            mask[0, 0, box_info[i][2]:box_info[i][2] + box_info[i][5], box_info[i][0]:box_info[i][0] + box_info[i][4]] = 1.0
            device = mask.device
            mask = mask.type(torch.FloatTensor).to(device)

            mask_sum_for_pred = torch.clamp(mask_sum_for_pred + mask, 0.0, 1.0)

            mask_list.append(pred_mask)
            featur_map_list.append(tmp_crop)

        pred_bg_mask = self.simple_bg_conv(bg_feature)
        mask_list.append(pred_bg_mask + (1 - mask_sum_for_pred) * 100000.0)
        mask_list = self.normalize(torch.cat(mask_list, 1))

        mask_list_maskout = mask_list.clone()
        
        instance_mask = torch.clamp(torch.sum(mask_list_maskout[:, :instance_feature.shape[0]], 1, keepdim=True), 0.0, 1.0)

        featur_map_list.append(bg_feature)
        featur_map_list = torch.cat(featur_map_list, 0)
        mask_list_maskout = mask_list_maskout.permute(1, 0, 2, 3).contiguous()
        out = featur_map_list * mask_list_maskout
        out = torch.sum(out, 0, keepdim=True)
        return out # , instance_mask, torch.clamp(mask_list, 0.0, 1.0)


class InstanceGenerator(nn.Module):
    '''
    Class for instance (bounding box images) image colorization.
    '''
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(InstanceGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Encoder
        self.model1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(64),

        )

        self.model2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(128),
        )

        self.model3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(256),

        )

        self.model4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(512),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Mu and Var
        self.model5_mu = nn.Linear(512 * 32 * 32, 64)
        self.model5_logvar = nn.Linear(512 * 32 * 32, 64)

        # Decoder
        self.decoder_input = nn.Linear(64, 512 * 32 * 32)

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(512),
        )

        # Here we are both deconvolving AND using a U-Net style skip connection from layer 3
        self.model7_upscale = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),) 
        self.model7_skip_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.model7 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(256),
        )

        self.model8_upscale = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),)
        self.model8_skip_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.model8 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            norm_layer(128),
        )

        self.model9_upscale = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias),)
        self.model9_skip_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),)

        self.model9 = nn.Sequential(
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(negative_slope=.2),
        )

        self.model_out = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),nn.Tanh())
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def forward(self, input_A, input_B, mask_B):
        init_image = torch.cat((input_A, input_B, mask_B), dim=1)

        conv1_2 = self.model1(init_image)
        conv2_3 = self.model2(conv1_2)
        conv3_4 = self.model3(conv2_3)
        conv4_5 = self.model4(conv3_4)

        mu = self.model5_mu(conv4_5.view(-1, 512 * 32 * 32))
        logvar = self.model5_logvar(conv4_5.view(-1, 512 * 32 * 32))

        # z = self.reparameterize(mu, logvar)

        # decoder_input = self.decoder_input(z)
        # decoder_input = decoder_input.view(-1, 512, 32, 32)

        conv6_7 = self.model6(conv4_5)
        conv7_up = self.model7_upscale(conv6_7) + self.model7_skip_3(conv3_4)
        conv7_8 = self.model7(conv7_up)
        conv8_up = self.model8_upscale(conv7_8) + self.model8_skip_2(conv2_3)
        conv8_9 = self.model8(conv8_up)
        conv9_up = self.model9_upscale(conv8_9) + self.model9_skip_1(conv1_2)
        conv9_10 = self.model9(conv9_up)
        out_reg = self.model_out(conv9_10)

        feature_map = {}

        feature_map['conv1_2'] = conv1_2
        feature_map['conv2_3'] = conv2_3
        feature_map['conv3_4'] = conv3_4
        feature_map['conv4_5'] = conv4_5
        feature_map['conv6_7'] = conv6_7
        feature_map['conv7_8'] = conv7_8
        feature_map['conv8_9'] = conv8_9
        feature_map['conv9_10'] = conv9_10

        feature_map['conv7_up'] = conv7_up
        feature_map['conv8_up'] = conv8_up
        feature_map['conv9_up'] = conv9_up

        feature_map['out_reg'] = out_reg

        # feature_map['mu'] = mu
        # feature_map['logvar'] = logvar
        # feature_map['decoder_input'] = decoder_input

        return (out_reg, feature_map, mu, logvar)