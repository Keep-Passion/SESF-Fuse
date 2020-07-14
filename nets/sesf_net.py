import os
import torch
import torch.nn as nn
import numpy as np
import skimage
import PIL.Image
import torch.nn.functional as f
import torchvision.transforms as transforms
from skimage import morphology
from skimage.color import rgb2gray
from collections import OrderedDict
class SESF_Fuse():
    """
    Fusion Class
    """
    def __init__(self, attention = 'cse'):
        # initialize model
        self.device = "cuda:0"
        self.model = SESFuseNet(attention)
        self.model_path = os.path.join(os.getcwd(), "nets", "parameters", "lp+lssim_se_sf_net_times30.pkl")
        self.model.load_state_dict(torch.load(self.model_path, map_location={'cuda:3': 'cuda:0'}))
        self.model.to(self.device)
        self.model.eval()

        self.mean_value = 0.4500517361627943
        self.std_value = 0.26465333914691797
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([self.mean_value], [self.std_value])
        ])

        # initialize fusion setting
        self.kernel_radius = 5
        self.area_ratio = 0.01
        self.ks = 5
        self.gf_radius = 4
        self.eps = 0.1

    def fuse(self, img1, img2):
        ndim = img1.ndim
        if ndim == 2:
            img1_gray = img1
            img2_gray = img2
        else:
            img1_gray = rgb2gray(img1)
            img2_gray = rgb2gray(img2)

        img1_gray_pil = PIL.Image.fromarray(img1_gray)
        img2_gray_pil = PIL.Image.fromarray(img2_gray)
        img1_tensor = self.data_transforms(img1_gray_pil).unsqueeze(0).to(self.device)
        img2_tensor = self.data_transforms(img2_gray_pil).unsqueeze(0).to(self.device)

        dm = self.model.forward("fuse", img1_tensor, img2_tensor, kernel_radius=self.kernel_radius)
        # Morphology filter and Small region removal
        h, w = img1.shape[:2]
        se = skimage.morphology.disk(self.ks)  # 'disk' kernel with ks size for structural element
        dm = skimage.morphology.binary_opening(dm, se)
        dm = morphology.remove_small_holes(dm == 0, self.area_ratio * h * w)
        dm = np.where(dm, 0, 1)
        dm = skimage.morphology.binary_closing(dm, se)
        dm = morphology.remove_small_holes(dm == 1, self.area_ratio * h * w)
        dm = np.where(dm, 1, 0)
        # guided filter
        if ndim == 3:
            dm = np.expand_dims(dm, axis=2)

        temp_fused = img1 * dm + img2 * (1 - dm)
        dm = self.guided_filter(temp_fused, dm, self.gf_radius, eps=self.eps)
        fused = img1 * 1.0 * dm + img2 * 1.0 * (1 - dm)
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return fused

    @staticmethod
    def box_filter(imgSrc, r):
        """
        Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
        :param imgSrc: np.array, image
        :param r: int, radius
        :return: imDst: np.array. result of calculation
        """
        if imgSrc.ndim == 2:
            h, w = imgSrc.shape[:2]
            imDst = np.zeros(imgSrc.shape[:2])

            # cumulative sum over h axis
            imCum = np.cumsum(imgSrc, axis=0)
            # difference over h axis
            imDst[0: r+1] = imCum[r: 2 * r+1]
            imDst[r + 1: h - r] = imCum[2 * r + 1: h] - imCum[0: h - 2 * r - 1]
            imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

            # cumulative sum over w axis
            imCum = np.cumsum(imDst, axis=1)

            # difference over w axis
            imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
            imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
            imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r]) - \
                                 imCum[:, w - 2 * r - 1: w - r - 1]
        else:
            h, w = imgSrc.shape[:2]
            imDst = np.zeros(imgSrc.shape)

            # cumulative sum over h axis
            imCum = np.cumsum(imgSrc, axis=0)
            # difference over h axis
            imDst[0: r + 1] = imCum[r: 2 * r + 1]
            imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - imCum[0: h - 2 * r - 1, :]
            imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

            # cumulative sum over w axis
            imCum = np.cumsum(imDst, axis=1)

            # difference over w axis
            imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
            imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
            imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - \
                                 imCum[:, w - 2 * r - 1: w - r - 1]
        return imDst

    def guided_filter(self, I, p, r, eps=0.1):
        """
        Guided Filter
        :param I: np.array, guided image
        :param p: np.array, input image
        :param r: int, radius
        :param eps: float
        :return: np.array, filter result
        """
        h, w = I.shape[:2]
        if I.ndim == 2:
            N = self.box_filter(np.ones((h, w)), r)
        else:
            N = self.box_filter(np.ones((h, w, 1)), r)
        mean_I = self.box_filter(I, r) / N
        mean_p = self.box_filter(p, r) / N
        mean_Ip = self.box_filter(I * p, r) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = self.box_filter(I * I, r) / N
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)

        if I.ndim == 2:
            b = mean_p - a * mean_I
            mean_a = self.box_filter(a, r) / N
            mean_b = self.box_filter(b, r) / N
            q = mean_a * I + mean_b
        else:
            b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
            mean_a = self.box_filter(a, r) / N
            mean_b = self.box_filter(b, r) / N
            q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
        return q


class SESFuseNet(nn.Module):
    """
    The Class of SESFuseNet
    """
    def __init__(self, attention = 'cse'):
        super(SESFuseNet, self).__init__()
        # Encode
        self.features = self.conv_block(in_channels=1, out_channels=16)
        self.conv_encode_1 = self.conv_block(16, 16)
        self.conv_encode_2 = self.conv_block(32, 16)
        self.conv_encode_3 = self.conv_block(48, 16)
        if(attention == 'cse'):
            self.se_f = CSELayer(16, 8)
            self.se_1 = CSELayer(16, 8)
            self.se_2 = CSELayer(16, 8)
            self.se_3 = CSELayer(16, 8)
        elif(attention == 'sse'):
            self.se_f = SSELayer(16)
            self.se_1 = SSELayer(16)
            self.se_2 = SSELayer(16)
            self.se_3 = SSELayer(16)
        elif(attention == 'scse'):
            self.se_f = SCSELayer(16, 8)
            self.se_1 = SCSELayer(16, 8)
            self.se_2 = SCSELayer(16, 8)
            self.se_3 = SCSELayer(16, 8)
        # Decode
        self.conv_decode_1 = self.conv_block(64, 64)
        self.conv_decode_2 = self.conv_block(64, 32)
        self.conv_decode_3 = self.conv_block(32, 16)
        self.conv_decode_4 = self.conv_block(16, 1)

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3):
        """
        The conv block of common setting: conv -> relu -> bn
        In conv operation, the padding = 1
        :param in_channels: int, the input channels of feature
        :param out_channels: int, the output channels of feature
        :param kernel_size: int, the kernel size of feature
        :return:
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
        return block

    @staticmethod
    def concat(f1, f2):
        """
        Concat two feature in channel direction
        """
        return torch.cat((f1, f2), 1)

    def forward(self, phase, img1, img2=None, kernel_radius=5):
        """
        Train or Forward for two images
        :param phase: str, 'train' or 'fuse'
        :param img1: torch.Tensor
        :param img2: torch.Tensor, only be used in 'fuse' mode
        :param kernel_radius: The kernel radius of spatial frequency
        :return: output, torch.Tensor
        """
        if phase == 'train':
            # Encode
            features = self.features(img1)
            se_features = self.se_f(features)
            encode_block1 = self.conv_encode_1(se_features)
            se_encode_block1 = self.se_1(encode_block1)
            se_cat1 = self.concat(se_features, se_encode_block1)
            encode_block2 = self.conv_encode_2(se_cat1)
            se_encode_block2 = self.se_2(encode_block2)
            se_cat2 = self.concat(se_cat1, se_encode_block2)
            encode_block3 = self.conv_encode_3(se_cat2)
            se_encode_block3 = self.se_3(encode_block3)
            se_cat3 = self.concat(se_cat2, se_encode_block3)
            # Decode
            decode_block1 = self.conv_decode_1(se_cat3)
            decode_block2 = self.conv_decode_2(decode_block1)
            decode_block3 = self.conv_decode_3(decode_block2)
            output = self.conv_decode_4(decode_block3)
        elif phase == 'fuse':
            with torch.no_grad():
                # Encode
                features_1 = self.features(img1)
                features_2 = self.features(img2)
                se_features_1 = self.se_f(features_1)
                se_features_2 = self.se_f(features_2)
                encode_block1_1 = self.conv_encode_1(se_features_1)
                encode_block1_2 = self.conv_encode_1(se_features_2)
                se_encode_block1_1 = self.se_1(encode_block1_1)
                se_encode_block1_2 = self.se_1(encode_block1_2)
                se_cat1_1 = self.concat(se_features_1, se_encode_block1_1)
                se_cat1_2 = self.concat(se_features_2, se_encode_block1_2)
                encode_block2_1 = self.conv_encode_2(se_cat1_1)
                encode_block2_2 = self.conv_encode_2(se_cat1_2)
                se_encode_block2_1 = self.se_2(encode_block2_1)
                se_encode_block2_2 = self.se_2(encode_block2_2)
                se_cat2_1 = self.concat(se_cat1_1, se_encode_block2_1)
                se_cat2_2 = self.concat(se_cat1_2, se_encode_block2_2)
                encode_block3_1 = self.conv_encode_3(se_cat2_1)
                encode_block3_2 = self.conv_encode_3(se_cat2_2)
                se_encode_block3_1 = self.se_3(encode_block3_1)
                se_encode_block3_2 = self.se_3(encode_block3_2)
                se_cat3_1 = self.concat(se_cat2_1, se_encode_block3_1)
                se_cat3_2 = self.concat(se_cat2_2, se_encode_block3_2)
            # SESF-Fuse calculate activity level
            output = self.fusion_channel_sf(se_cat3_1, se_cat3_2, kernel_radius=kernel_radius)
        return output

    # gpu mode
    @staticmethod
    def fusion_channel_sf(f1, f2, kernel_radius=5):
        """
        Perform channel sf fusion two features
        """
        device = f1.device
        b, c, h, w = f1.shape
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = f.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = f.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = f.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = f.conv2d(f2, b_shift_kernel, padding=1, groups=c)

        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)

        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
        kernel_padding = kernel_size // 2
        f1_sf = torch.sum(f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
        f2_sf = torch.sum(f.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
        weight_zeros = torch.zeros(f1_sf.shape).cuda(device)
        weight_ones = torch.ones(f1_sf.shape).cuda(device)

        # get decision map
        dm_tensor = torch.where(f1_sf > f2_sf, weight_ones, weight_zeros).cuda(device)
        dm_np = dm_tensor.squeeze().cpu().numpy().astype(np.int)
        return dm_np

class CSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SSELayer(nn.Module):
    def __init__(self, channel):
        super(SSELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.fc(x) 
        return x * y 
class SCSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSELayer, self).__init__()
        self.CSE = CSELayer(channel, reduction=reduction)
        self.SSE = SSELayer(channel)

    def forward(self, U):
        SSE = self.SSE(U)
        CSE = self.CSE(U)
        return SSE+CSE
