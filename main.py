import os
from skimage import io
from nets.sesf_net import SESF_Fuse


def main(input_dir, output_dir):
    """
    Image Fusion
    :param input_dir: str, input dir with all images stores in one folder
    :param output_dir: str, output dir with all fused images
    :return:
    """
    sesf = SESF_Fuse("cse")
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image_name in images_name:
        print("Fusing {}".format(image_name))
        img1 = io.imread(os.path.join(input_dir, image_name + "_1.png"))
        img2 = io.imread(os.path.join(input_dir, image_name + "_2.png"))
        fused = sesf.fuse(img1, img2)
        io.imsave(os.path.join(output_dir, image_name + ".png"), fused)


if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), "data", "multi_focus")
    output_dir = os.path.join(os.getcwd(), "data", "result")
    main(input_dir, output_dir)
