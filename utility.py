import os
import shutil
import cv2
import matlab
import matlab.engine
import skimage.transform

def make_out_dir(dir_path):
    """
    create a folder
    :param dir_path: the address of folder
    :return: None
    """
    try:
        os.makedirs(dir_path)
    except OSError:
        pass


def delete_folder(dir_address):
    """
    delete all the files in folder and the folder
    :param dir_address: the address of folder
    :return: None
    """
    shutil.rmtree(dir_address)


def resize_image(origin_image, resize_times, inter_method=cv2.INTER_CUBIC):
    """
    Rescaling image with certain ratio, we have tes opencv and skimage, it is the better
    :param origin_image:np.array
    :param resize_times: ratio, float
    :param inter_method: cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC,cv2.INTER_TAB_SIZE2
    :return: result_image,np.array
    """
    (h, w) = origin_image.shape[:2]
    resize_h = int(h * resize_times)
    resize_w = int(w * resize_times)
    result_image = cv2.resize(origin_image, (resize_w, resize_h), interpolation=inter_method)
    return result_image


class MatlabEngine():
    """
    Matlab Engine Class, used for python incorporate with matlab
    """
    def __init__(self):
        project_address = os.getcwd()
        self.matlab_engine = matlab.engine.start_matlab()
        # Add the search dir of matlab
        self.matlab_engine.addpath(os.path.join(project_address, 'evaluation_methods'))
        self.matlab_engine.addpath(os.path.join(project_address, 'evaluation_methods', "matlabPyrTools"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "dsift"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "gf"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "mwg"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "ifm"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "ifm", "mattingtoolbox"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "MST_SR_fusion_toolbox"))
        self.matlab_engine.addpath(os.path.join(project_address, "fusion_methods", "cnn_fusion"))
        self.matlab_engine.addpath(os.path.join(project_address,
                                                "fusion_methods", "MST_SR_fusion_toolbox", "dtcwt_toolbox"))
        self.matlab_engine.addpath(os.path.join(project_address,
                                                "fusion_methods","MST_SR_fusion_toolbox",
                                                "fdct_wrapping_matlab"))
        self.matlab_engine.addpath(os.path.join(project_address,
                                                "fusion_methods", "MST_SR_fusion_toolbox", "nsct_toolbox"))
        self.matlab_engine.addpath(os.path.join(project_address,
                                                "fusion_methods", "MST_SR_fusion_toolbox", "sparsefusion"))
        self.matlab_engine.addpath(os.path.join(project_address,
                                                "fusion_methods","MST_SR_fusion_toolbox","sparsefusion",
                                                "ksvdbox"))
        self.matlab_engine.addpath(os.path.join(project_address,
                                                "fusion_methods","MST_SR_fusion_toolbox","sparsefusion",
                                                "ksvdbox","ompbox"))

    def __del__(self):
        self.matlab_engine.quit()


class LogInformation:
    """
    Log class to record information of experiment
    """
    def __init__(self, is_print_screen=True, is_out_log_file=True, log_file_address="log.txt"):
        """
        Determine the parameters of Log class
        :param is_print_screen: Bool
        :param is_out_log_file: Bool
        :param log_file_address: str
        """
        self.is_print_screen = is_print_screen
        self.is_out_log_file = is_out_log_file
        self.log_file_address = log_file_address
        if os.path.exists(self.log_file_address):
            os.remove(self.log_file_address)

    def print_and_log(self, content, is_end_blank=True):
        """
        print and log content
        :param content: str
        :param is_end_blank: BOOL, whether to blank in the end
        :return: None
        """
        if self.is_print_screen:
            if is_end_blank:
                print(content)
            else:
                print(content, end="")
        if self.is_out_log_file:
            f = open(self.log_file_address, "a")
            f.write(content)
            if is_end_blank:
                f.write("\n")
            f.close()