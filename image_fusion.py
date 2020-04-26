import numpy as np
import matlab
import matlab.engine
from skimage.color import rgb2gray
from scipy.io import loadmat
from fusion_methods.focus_stack import focus_stack
from fusion_methods.dense_fuse import dense_fuse_forward
from fusion_methods.fusion_gan import fusion_gan_forward
from fusion_methods.deep_fuse import deep_fuse_forward
from fusion_methods.sesf_fuse import sesf_fuse_forward
from utility import *


class ImageFusion(MatlabEngine):
    """
    Fusion class
    """
    def __init__(self):
        super(ImageFusion, self).__init__()
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    def __del__(self):
        super(ImageFusion, self).__del__()

    @staticmethod
    def _np_to_mat(img):
        """
        Transfer numpy to matlab style
        :param img: image, np.array
        :return: matlab style
        """
        img_mat = matlab.double(img.tolist())
        return img_mat

    @staticmethod
    def _mat_to_np(img_mat):
        """
        Transfer matlab style to numpy
        :param img: image, matlab style
        :return: np.array
        """
        img_np = np.array(img_mat)
        return img_np

    # ****************************************** DSIFT ***********************************************************
    def fuse_by_dense_sift(self, img1, img2, scale=48.0, block_size=8.0, matching=1.0):
        """
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        Liu Y, Liu S, Wang Z. Multi-focus image fusion with dense SIFT[J]. Information Fusion, 2015, 23: 139-155.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :param scale: used for disft calculation
        :param block_size: used for sliding window
        :param matching: whether to use matching process, if 1, it means used.
        :return: fused result, np.array
        """
        img1_mat = self._np_to_mat(img1)
        img2_mat = self._np_to_mat(img2)
        fused_mat = self.matlab_engine.DSIFT_Fusion(img1_mat, img2_mat, scale, block_size, matching)
        fused = self._mat_to_np(fused_mat).astype(np.uint8)
        return fused

    # ****************************************** Focus Stack *****************************************************
    def fuse_by_focus_stack(self, img1, img2):
        """
        Focus Stack image Fusion
        The codes are copied from https://github.com/cmcguinness/focusstack
        For those who are not familiar with focus stacking,
        this Wikipedia article (https://en.wikipedia.org/wiki/Focus_stacking)
        does a nice job of explaining the idea.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = focus_stack([img1, img2])
        return fused

    # ****************************************** Spatial Frequency ***********************************************
    def fuse_by_sf(self, img1, img2):
        """
        Spatial Frequency Image fusion
        Li S , Kwok J T , Wang Y . Combination of images with diverse focuses using the spatial frequency[J].
        Information Fusion, 2001, 2(3):169-176.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fusion result, np.array
        """
        weight_matrix = self._get_spatial_frequency_matrix(img1, img2)
        if img1.ndim == 3:
            weight_matrix = np.expand_dims(weight_matrix, axis=2)
        fuse_region = img1 * weight_matrix + img2 * (1 - weight_matrix)
        fuse_region = np.clip(fuse_region, 0, 255).astype(np.uint8)
        return fuse_region

    def _get_spatial_frequency_matrix(self, img1, img2, block_size=7, th=0.5):
        """
        Get weight matrix by using spatial_frequency
        :param img1: last image, np.array
        :param img2: next image, np.array
        :param block_size: used for calculate sf in a block
        :param th: used for calculate fused image in formula(4) in paper
        :return: weight matrix, np.array
        """
        if img1.ndim == 3:            # Convert to Gray mode, note rgb in skimage and matlab or gbr in cv2
            img1 = rgb2gray(img1)
            img2 = rgb2gray(img2)
            # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        row, col = img1.shape[0:2]
        row_num = row // block_size
        col_num = col // block_size
        row_have_remain = False
        col_have_remain = False
        if row % block_size != 0:
            row_have_remain = True
        if col % block_size != 0:
            col_have_remain = True
        weight_matrix = np.ones((row, col))  # decision map
        for i in range(row_num + 1):
            for j in range(col_num + 1):
                row_end_position = (i + 1) * block_size
                col_end_position = (j + 1) * block_size
                if i == row_num and row_have_remain is False:
                    continue
                elif i == row_num and row_have_remain:
                    row_end_position = row
                if j == col_num and col_have_remain is False:
                    continue
                elif j == col_num and col_have_remain:
                    col_end_position = col
                img1_block = img1[(i * block_size):row_end_position, (j * block_size):col_end_position]
                img2_block = img2[(i * block_size):row_end_position, (j * block_size):col_end_position]
                img1_block_sf = self._calculate_sf(img1_block)
                img2_block_sf = self._calculate_sf(img2_block)
                # if img1_block_sf >= img2_block_sf: # img1 is more clarify than img2 in block, set 1
                #     weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = \
                #         np.ones((row_end_position - (i * block_size), col_end_position - (j * block_size)))
                # else:  # img2 is more clarify than img1 in block, set 0
                #     weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = \
                #         np.zeros((row_end_position - (i * block_size), col_end_position - (j * block_size)))
                if img1_block_sf > img2_block_sf + th:    # img1 is more clarify than img2 in block, set 1
                    weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = 1.0
                elif img1_block_sf < img2_block_sf - th:  # img2 is more clarify than img1 in block, set 0
                    weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = 0.0
                else:                                     # uncertainty, set 0.5
                    weight_matrix[(i * block_size):row_end_position, (j * block_size):col_end_position] = 0.5
        # Verify and correct the weight matrix
        weight_matrix = self._verify_weight_matrix(weight_matrix)
        return weight_matrix

    @staticmethod
    def _calculate_sf(img):
        """
        Calculate spatial frequency
        :param img, np.array
        :return: sf, float
        """
        rf_temp = 0
        cf_temp = 0
        row, col = img.shape[0:2]
        img_temp = img.astype(int)
        for i in range(row):
            for j in range(col):
                if j < col - 1:
                    rf_temp = rf_temp + np.square(img_temp[i, j + 1] - img_temp[i, j])
                if i < row - 1:
                    cf_temp = cf_temp + np.square(img_temp[i + 1, j] - img_temp[i, j])
        rf = np.sqrt(float(rf_temp) / float(row * col))
        cf = np.sqrt(float(cf_temp) / float(row * col))
        sf = np.sqrt(np.square(rf) + np.square(cf))
        return sf

    @staticmethod
    def _verify_weight_matrix(weight_matrix):
        """
        Verify and correct the weight matrix by using majority principle with 3 * 3 window
        :param weight_matrix, np.array
        :return: verify_matrix, np.array
        """
        verify_matrix = weight_matrix.copy()
        h, w = verify_matrix.shape[:2]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if verify_matrix[i, j] == 0.5:
                    continue
                temp_sum = verify_matrix[i - 1, j - 1] + verify_matrix[i, j - 1] + verify_matrix[i + 1, j - 1] +\
                           verify_matrix[i - 1, j] + verify_matrix[i, j] + verify_matrix[i + 1, j] + \
                           verify_matrix[i - 1, j + 1] + verify_matrix[i, j + 1] + verify_matrix[i + 1, j + 1]
                if verify_matrix[i, j] == 0 and temp_sum == 8:
                    verify_matrix[i, j] = 1
                elif verify_matrix[i, j] == 1 and temp_sum == 0:
                    verify_matrix[i, j] = 0
        return verify_matrix

    # ****************************************** Guided Filtering ***********************************************
    def fuse_by_gf(self, img1, img2):
        """
        Guided filtering image Fusion
        The codes are copied from http://xudongkang.weebly.com/index.html
        Li S, Kang X, Hu J. Image fusion with guided filtering[J].
        IEEE Transactions on Image processing, 2013, 22(7): 2864-2875.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        # The script of GF need to convert np.array to the form of [h,w, image_num] or [h,w,3,image_num], so we add
        # dimension
        img1_expand = np.expand_dims(img1, img1.ndim)
        img2_expand = np.expand_dims(img2, img2.ndim)
        img = np.concatenate((img1_expand, img2_expand), axis=img1_expand.ndim - 1)
        img_mats = self._np_to_mat(img)
        fused_mat = self.matlab_engine.GFF(img_mats)
        fused = np.clip(self._mat_to_np(fused_mat), 0, 255).astype(np.uint8)
        return fused

    # ************************************************* NSCT ****************************************************
    def fuse_by_nsct(self, img1, img2):
        """
        NSCT （nonsubsampled contourlet transform）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        Q. Zhang, B. Guo, Multifocus image fusion using the nonsubsampled contourlet
        transform, Signal Process. 89 (7) (2009) 1334–1346
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        param_mat = self._np_to_mat(np.array([2,3,3,4]))
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.nsct_fuse(img1_mat, img2_mat, param_mat)
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.nsct_fuse(img1_mat, img2_mat, param_mat)
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    # ************************************************* CVT *****************************************************
    def fuse_by_cvt(self, img1, img2):
        """
        CVT （Curvelet transform）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        F. Nencini, A. Garzelli, S. Baronti, L. Alparone, Remote sensing image fusion
        using the curvelet transform, Inform. Fusion 8 (2) (2007) 143–156
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        level = 5.0
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.curvelet_fuse(img1_mat, img2_mat, level)
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.curvelet_fuse(img1_mat, img2_mat, level)
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    # ************************************************* DWT *****************************************************
    def fuse_by_dwt(self, img1, img2):
        """
        DWT （Discrete wavelet transform）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        H. Li, B. Manjunath, S. Mitra, Multisensor image fusion using the wavelet
        transform, Graph. Models Image Process. 57 (3) (1995) 235–245.
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        h, w = img1.shape[:2]
        # The dwt acquire imge with shape of 2X
        # So we resize image with odd shape first and resize back
        have_odd_shape = False
        if h % 2 or w % 2:
            have_odd_shape = True
        if have_odd_shape:
            img1 = self.resize_odd_to_even(img1)
            img2 = self.resize_odd_to_even(img2)
        # Because dwt need image to perform multiple decomposition
        # we need first the level of decomposition
        level = self.find_suit_level(img1)
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            temp_fused = self.matlab_engine.dwt_fuse(img1_mat, img2_mat, level)
            temp_fused = np.array(temp_fused)
            if have_odd_shape:
                temp_fused = cv2.resize(temp_fused, (w, h), interpolation=cv2.INTER_AREA)
            fused = temp_fused
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                temp_fused = self.matlab_engine.dwt_fuse(img1_mat, img2_mat, level)
                temp_fused = np.array(temp_fused)
                if have_odd_shape:
                    temp_fused = cv2.resize(temp_fused, (w, h), interpolation=cv2.INTER_AREA)
                fused[:, :, index] = temp_fused
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    @staticmethod
    def resize_odd_to_even(img):
        """
        Resize the img with odd shape to even
        We simply add 1 for odd shape
        :param img: np.array
        :return: resize_img, np.array
        """
        h, w = img.shape[:2]
        have_odd_h = have_odd_w = False
        if h % 2:
            have_odd_h = True
        if w % 2:
            have_odd_w = True
        resize_h = h
        resize_w = w
        if have_odd_h:
            resize_h = resize_h + 1
        if have_odd_w:
            resize_w = resize_w + 1
        resize_img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        # resize_img = skimage.transform.resize(img, (resize_h, resize_w),
                                              # mode='reflect', preserve_range=True, anti_aliasing=True)
        return resize_img

    @staticmethod
    def find_suit_level(img):
        """
        Find suit level for decomposition operation
        :param img: np.array
        :return: level, float
        """
        h, w = img.shape[:2]
        level = 1.0
        while h % 2 or w % 2:
            level = level + 1
            h = h / 2
            w = w / 2
        return level

    # ************************************************* DTCWT ***************************************************
    def fuse_by_dtcwt(self, img1, img2):
        """
        DTCWT image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        J. Lewis, R. OCallaghan, S. Nikolov, D. Bull, N. Canagarajah, Pixel- and regionbased
        image fusion with complex wavelets, Inform. Fusion 8 (2) (2007) 119–130.
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        h, w =img1.shape[:2]
        fused = np.zeros(img1.shape)
        have_odd_shape = False
        if h % 2 or w % 2:
            have_odd_shape = True
        if have_odd_shape:
            img1 = self.resize_odd_to_even(img1)
            img2 = self.resize_odd_to_even(img2)
        level = 4.0
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            temp_fused = self.matlab_engine.dtcwt_fuse(img1_mat, img2_mat, level)
            temp_fused = np.array(temp_fused)
            if have_odd_shape:
                temp_fused = cv2.resize(temp_fused, (w, h), interpolation=cv2.INTER_AREA)
            fused = temp_fused
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                temp_fused = self.matlab_engine.dtcwt_fuse(img1_mat, img2_mat, level)
                temp_fused = np.array(temp_fused)
                if have_odd_shape:
                    temp_fused = cv2.resize(temp_fused, (w, h), interpolation=cv2.INTER_AREA)
                fused[:, :, index] = temp_fused
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    # ************************************************** LP *****************************************************
    def fuse_by_lp(self, img1, img2):
        """
        LP （Laplacian pyramid）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        P. Burt, E. Adelson, The laplacian pyramid as a compact image code, IEEE Trans.
        Commun. 31 (4) (1983) 532–540.
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        level = 4
        ap = mp = 3
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.lp_fuse(img1_mat, img2_mat, level, ap, mp)
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.lp_fuse(img1_mat, img2_mat, level, ap, mp)
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    # ************************************************** RP *****************************************************
    def fuse_by_rp(self, img1, img2):
        """
        RP （Ratio of low-pass pyramid）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        A. Toet, Image fusion by a ratio of low pass pyramid, Pattern Recogn. Lett. 9 (4)
        (1989) 245–253
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        level = 4
        ap = mp = 3
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.rp_fuse(img1_mat, img2_mat, level, ap, mp)
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.rp_fuse(img1_mat, img2_mat, level, ap, mp)
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    # ************************************************** SR *****************************************************
    def fuse_by_sr(self, img1, img2):
        """
        SR （Sparse representation）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        B. Yang, S. Li, Multifocus image fusion and restoration with sparse representation,
        IEEE Trans. Instrum. Meas. 59 (4) (2010) 884–892.
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        pwd = os.getcwd()
        file_address = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(pwd, "fusion_methods"),
                                    "MST_SR_fusion_toolbox"),
                                    "sparsefusion"),
                                    "Dictionary"),
                                    "D_100000_256_8.mat")

        dict_mat = self._np_to_mat(loadmat(file_address)['D'])
        overlap = 6
        epsilon = 0.1
        if img1.ndim == 2:     # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.sparse_fusion(img1_mat, img2_mat, dict_mat, overlap, epsilon)
        else:                        # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.sparse_fusion(img1_mat, img2_mat, dict_mat, overlap, epsilon)
        fused = np.clip(self._mat_to_np(fused), 0, 255).astype(np.uint8)
        return fused

    # ************************************************** MWG ****************************************************
    def fuse_by_mwg(self, img1, img2):
        """
        MWG image fusion
        The codes are copied from https://github.com/lsauto/MWGF-Fusion
        Zhou Z, Li S, Wang B. Multi-scale weighted gradient-based fusion for multi-focus images[J].
        Information Fusion, 2014, 20: 60-72.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        img1_mat = self._np_to_mat(img1)
        img2_mat = self._np_to_mat(img2)
        fused_mat = self.matlab_engine.MWGFusion(img1_mat, img2_mat)
        fused = np.clip(self._mat_to_np(fused_mat), 0, 255).astype(np.uint8)
        return fused

    # ************************************************** IMF ****************************************************
    def fuse_by_imf(self, img1, img2):
        """
        IMF (Image Matting) image fusion
        The codes are copied from http://xudongkang.weebly.com/index.html
        Li S,   Kang X, Hu J, et al. Image matting for fusion of multi-focus images in dynamic scenes[J].
        Information Fusion, 2013, 14(2): 147-162.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        # The script of imf need to convert np.array to the form of [h,w, image_num] or [h,w,3,image_num], so we add
        # dimension
        img1_expand = np.expand_dims(img1, img1.ndim)
        img2_expand = np.expand_dims(img2, img1.ndim)
        img = np.concatenate((img1_expand, img2_expand), axis=img1_expand.ndim - 1)
        img_mats = self._np_to_mat(img)
        fused_mat = self.matlab_engine.IFM(img_mats)
        fused = np.clip(self._mat_to_np(fused_mat), 0, 255).astype(np.uint8)
        return fused

    # ********************************************** Dense Fuse *************************************************
    @staticmethod
    def fuse_by_dense_fuse_1e3_add(img1, img2):
        """
        densefuse
        The codes are copied from https://github.com/hli1221/imagefusion_densefuse
        H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,”
        IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        return dense_fuse_forward(img1, img2, ssim_weight_index=3, type="add")

    @staticmethod
    def fuse_by_dense_fuse_1e3_l1(img1, img2):
        """
        densefuse
        The codes are copied from https://github.com/hli1221/imagefusion_densefuse
        H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,”
        IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        return dense_fuse_forward(img1, img2, ssim_weight_index=3, type="l1")

    # ********************************************** CNN Fuse ***************************************************
    def fuse_by_cnn(self, img1, img2):
        """
        cnn fuse
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        Yu Liu, Xun Chen, Hu Peng, Zengfu Wang, Multi-focus image fusion with a deep
        convolutional neural network, Information Fusion, 36: 191-207, 2017.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        img1_mat = self._np_to_mat(img1)
        img2_mat = self._np_to_mat(img2)
        fused_mat = self.matlab_engine.CNN_Fusion(img1_mat, img2_mat, "cnnmodel.mat")
        fused = np.clip(self._mat_to_np(fused_mat), 0, 255).astype(np.uint8)
        return fused

    # ********************************************** Deep Fuse **************************************************
    def fuse_by_deep_fuse(self, img1, img2):
        """
        deep fuse
        The codes are copied from https://github.com/SunnerLi/DeepFuse.pytorch
        K. R. Prabhakar, V. S. Srikar, and R. V. Babu. Deepfuse: A deep unsupervised approach for exposure fusion with extreme exposure image pairs. In 2017 IEEE International Conference on Computer Vision (ICCV).
        IEEE, pages 4724–4732, 2017.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        cwd = os.getcwd()
        pre_train_model_path = os.path.join(cwd, "fusion_methods", "deep_fuse_models", "model", "best_model.pth")
        return deep_fuse_forward(img1, img2, pre_train_model_path)

    # ********************************************** Fusion Gan *************************************************
    def fuse_by_fuison_gan(self, img1, img2):
        """
        Fusion Gan
        The codes are copied from https://github.com/jiayi-ma/FusionGAN
        Jiayi Ma, Wei Yu, Pengwei Liang, Chang Li, and Junjun Jiang. "FusionGAN: A generative adversarial network for infrared and visible image fusion",
        Information Fusion, 48, pp. 11-26, Aug. 2019.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        cwd = os.getcwd()
        reader_path = os.path.join(cwd, "fusion_methods", "fusion_gan", "checkpoint", "CGAN_120", "CGAN.model-3")
        return fusion_gan_forward(img1, img2, reader_path)

    # ********************************************** SESF Fuse **************************************************
    def fuse_by_sesf_fuse(self, img1, img2, fuse_type="se_sf_dm"):
        """
        SESF-Fusion
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        :return:
        """
        return sesf_fuse_forward(img1, img2, fuse_type=fuse_type)
