import os
import numpy as np
import pandas as pd
from skimage import io
from image_fusion import ImageFusion
from evaluation import Evaluator
from utility import *


# We provide these fusion methods, one can use it by using the index
# 19,20,21,22,23,24 are used for ablation experiment in our paper
methods_name = ['gf', 'dsift', 'focus_stack', 'sf',                                                           # 0,1,2,3
                'nsct', 'cvt', 'dwt', 'lp',                                                                    # 4,5,6,7
                'rp', 'dtcwt', "sr", "mwg", 'imf',                                                            # 8,9,10.11,12
                'cnn_fuse', 'dense_fuse_1e3_add', 'dense_fuse_1e3_l1', 'deep_fuse', "fusion_gan",        # 13,14,15,16,17
                'se_sf_dm', 'dense_sf_dm', 'se_sf', 'se_average', 'se_max', 'se_absmax', 'se_l1_norm',    # 18,19,20,21,22,23,24
                'sse_sf_dm','scse_sf_dm']                                                           # 25,26


# We provide these metrics, one can use it by using the index
metrics_name = ['Qmi', 'Qte', 'Qncie','Qg', 'Qm', 'Qsf', 'Qp','Qs', 'Qc', 'Qy','Qcv', 'Qcb',
                'VIFF','MEF_SSIM','SSIM_A','FMI_EDGE', 'FMI_DCT', 'FMI_W','Nabf','SCD','SD','SF', 'CC']


def fuse_images(input_dir, output_dir, methods_id=[0, 1, 2, 3], log_address="log.txt"):
    """
    Fuse multi-focus images with different methods evaluated by different metrics
    :param input_dir: data dir, str, the data need to be grouped in the sub-folder
    :param output_dir: result dir, str, the function will create two sub-folder,"different_methods" and "record"
    :param methods_id: list int, the ids of fuse method
    :param log_address: str, the log address of result
    :return: None
    """
    image_fusion = ImageFusion()
    logs = LogInformation(log_file_address=log_address, is_print_screen=True, is_out_log_file=True)
    # images_name = os.listdir(input_dir)
    # Demonstrating fusing method
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    logs.print_and_log("The input dir is: {}".format(input_dir))
    logs.print_and_log("The output dir is: {}".format(output_dir))
    logs.print_and_log("The fuse methods in this experiment are:")
    for item in methods_id:
        logs.print_and_log(" " + methods_name[item], is_end_blank=False)
    logs.print_and_log("")  # blank
    logs.print_and_log("Start Fusing")
    for method_index, method_id in enumerate(methods_id):  # fused by different methods:
        logs.print_and_log("Fusing method: {}".format(methods_name[method_id]))
        for image_index, image_name in enumerate(images_name):  # read every image which need to be fused
            if not image_name.startswith('.'):
                logs.print_and_log("Analysis {}".format(image_name))
                # image_dir = os.path.join(input_dir, image_name)
                img1 = io.imread(os.path.join(input_dir, image_name + "_1.png"))
                img2 = io.imread(os.path.join(input_dir, image_name + "_2.png"))
                assert img1.shape == img2.shape, "The two images have different shapes"
                if method_id == 0:     # Guided Filtering(GF)
                    fused = image_fusion.fuse_by_gf(img1, img2)
                elif method_id == 1:   # DSIFT
                    fused = image_fusion.fuse_by_dense_sift(img1, img2)
                elif method_id == 2:   # Focus Stack
                    fused = image_fusion.fuse_by_focus_stack(img1, img2)
                elif method_id == 3:   # Spatial Frequency(SF)
                    fused = image_fusion.fuse_by_sf(img1, img2)
                elif method_id == 4:   # NSCT
                    fused = image_fusion.fuse_by_nsct(img1, img2)
                elif method_id == 5:   # CVT
                    fused = image_fusion.fuse_by_cvt(img1, img2)
                elif method_id == 6:  # DWT
                    fused = image_fusion.fuse_by_dwt(img1, img2)
                elif method_id == 7:  # LP
                    fused = image_fusion.fuse_by_lp(img1, img2)
                elif method_id == 8:  # RP
                    fused = image_fusion.fuse_by_rp(img1, img2)
                elif method_id == 9:  # DTCWT
                    fused = image_fusion.fuse_by_dtcwt(img1, img2)
                elif method_id == 10:  # SR
                    fused = image_fusion.fuse_by_sr(img1, img2)
                elif method_id == 11:  # MWG
                    fused = image_fusion.fuse_by_mwg(img1, img2)
                elif method_id == 12:  # IMF
                    fused = image_fusion.fuse_by_imf(img1, img2)
                elif method_id == 13:  # CNN fuse
                    fused = image_fusion.fuse_by_cnn(img1, img2)
                elif method_id == 14:  # Dense fuse - 1e2 - addition
                    fused = image_fusion.fuse_by_dense_fuse_1e3_add(img1, img2)
                elif method_id == 15:  # Dense fuse- 1e2 - l1 strategy
                    fused = image_fusion.fuse_by_dense_fuse_1e3_l1(img1, img2)
                elif method_id == 16:  # Deep fuse
                    fused = image_fusion.fuse_by_deep_fuse(img1, img2)
                elif method_id == 17:  # Fusion Gan
                    # We recommend to only use in infrared and visible
                    # image fusion as the paper designed for
                    fused = image_fusion.fuse_by_fuison_gan(img1, img2)
                elif method_id == 18:  # SESF Fuse - se_sf_dm
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='se_sf_dm')
                    # Below experiments are used for ablation experiments in our paper
                elif method_id == 19:  # SESF Fuse - dense_sf_dm
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='dense_sf_dm')
                elif method_id == 20:  # SESF Fuse - se_sf
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='se_sf')
                elif method_id == 21:  # SESF Fuse - se_average
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='se_average')
                elif method_id == 22:  # SESF Fuse - se_max
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='se_max')
                elif method_id == 23:  # SESF Fuse - se_absmax
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='se_absmax')
                elif method_id == 24:  # SESF Fuse - se_l1_norm
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='se_l1_norm')
                elif method_id == 25:  # SESF Fuse - sse_sf
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='sse_sf_dm')
                elif method_id == 26:  # SESF Fuse - scse_sf
                    fused = image_fusion.fuse_by_sesf_fuse(img1, img2, fuse_type='scse_sf_dm')
                fused_dir = os.path.join(output_dir, "different_methods", methods_name[method_id])
                make_out_dir(fused_dir)
                io.imsave(os.path.join(fused_dir, image_name + ".png"), fused)
    logs.print_and_log("End Fusing, Please check output files")


def evaluate_fused(input_dir, fused_dir, record_dir, methods_list, metrics_id=[0,1,2,3,4,5,6,7,8,9,10,11,12], log_address="log.txt"):
    """
    Evaluate different fusion results by metrics
    :param input_dir: data dir, str, the data need to be grouped in the sub-folder
    :param fused_dir: fused dir, str, the fused result dir of different methods
    :param record_dir: evaluation dir, str, we output result in this dir
    :param methods_list: list, str, the methods need to be evaluated
    :param metrics_id: list, int, the id of metrics
    :param log_address: str, the log address of result
    :return: None
    """
    logs = LogInformation(log_file_address=log_address, is_print_screen=True, is_out_log_file=True)
    logs.print_and_log("The metrics in this experiment are:")
    for item in metrics_id:
        logs.print_and_log(" " + metrics_name[item], is_end_blank=False)
    logs.print_and_log("")  # blank
    evaluator = Evaluator()
    # image_names = os.listdir(input_dir)
    image_names = sorted(list({item[:-6] for item in os.listdir(input_dir)}))
    for method_name in methods_list:
        logs.print_and_log("Evaluating result for {}".format(method_name))
        fused_method_dir = os.path.join(fused_dir, method_name)
        record_address = os.path.join(record_dir, "evaluator_" + method_name + ".npy")
        # We save the result with the array in ".npy" file
        # The shape is (len(metric_ids), len(image_names))
        eval_result = np.zeros((len(metrics_id), len(image_names)))
        for image_index, image_name in enumerate(image_names):
            logs.print_and_log("Analyzing {}/{}, {}".format(image_index + 1, len(image_names), image_name))
            img1 = io.imread(os.path.join(input_dir, image_name + "_1.png"))
            img2 = io.imread(os.path.join(input_dir, image_name + "_2.png"))
            fused = io.imread(os.path.join(fused_method_dir, image_name + ".png"))
            metrics_list = evaluator.get_evaluation(img1, img2, fused, metric_ids=metrics_id)
            for metric_index, metric in zip(metrics_id, metrics_list):
                logs.print_and_log(" {}={:.4f}".format(metrics_name[metric_index], metric), is_end_blank=False)
            logs.print_and_log("")
            for metric_index, metric in enumerate(metrics_list):
                eval_result[metric_index, image_index] = metric
        np.save(record_address, eval_result)


def export_result(input_dir, output_dir, record_dir, records_list, metrics_id):
    """
    Export result in excel
    :param input_dir: data dir, str, the data need to be grouped in the sub-folder
    :param output_dir: the dir of output excel
    :param record_dir: evaluation dir, str
    :param records_list: the npy file that need to be export
    :return: None
    """
    images_name = sorted(list({item[:-6] for item in os.listdir(input_dir) if not item.startswith('.')}))
    # shape(method_number, metric_number, image_number)
    eval_result = np.zeros((len(records_list), len(metrics_id), len(images_name)))
    local_metrics_name = []
    for metric_id in metrics_id:
        local_metrics_name.append(metrics_name[metric_id])
    local_methods_name = []
    for method_index, evaluator_name in enumerate(records_list):
        local_methods_name.append(evaluator_name[10:-4])
        eval_result[method_index, :, :] = \
            np.load(os.path.join(record_dir, evaluator_name))[metrics_id, :]
    out_excel_address = os.path.join(output_dir, "result.xlsx")
    methods_number = len(local_methods_name)
    metrics_number = len(local_metrics_name)
    images_number = len(images_name)

    # total evaluate by average shape = (methods_number, metrics_number)
    total_eval = np.average(eval_result, axis=2)
    total_df = pd.DataFrame(total_eval)
    total_df.columns = local_metrics_name
    total_df.index = local_methods_name

    # total evaluate by first place number - shape = (methods_number, metrics_number)
    fp_eval = np.zeros((methods_number, metrics_number))
    for metric_index in range(0, metrics_number):
        for image_index in range(0, images_number):
            temp_value = np.max(eval_result[:, metric_index, image_index])
            for method_index in range(0, methods_number):
                if eval_result[method_index, metric_index, image_index] == temp_value:
                    fp_eval[method_index, metric_index] += 1
    fp_df = pd.DataFrame(fp_eval)
    fp_df.columns = local_metrics_name
    fp_df.index = local_methods_name

    # order - shape = (methods_number, metrics_number)
    total_order_eval = np.zeros(total_eval.shape)
    for metrics_index in range(metrics_number):
        temp_list = total_eval[:, metrics_index].tolist()
        order_list = sorted(temp_list, reverse=True)
        for method_index in range(methods_number):
            total_order_eval[method_index, metrics_index] = \
                order_list.index(total_eval[method_index, metrics_index]) + 1
    total_order_df = pd.DataFrame(total_order_eval)
    total_order_df.columns = local_metrics_name
    total_order_df.index = local_methods_name

    # detail evaluate by each metric shape = (methods_number, images_number)
    q_df = []
    q_order_df = []
    # eval_result.shape = (method_number, metric_number, image_number)
    for metric_index in range(metrics_number):
        temp = eval_result[:, metric_index, :]
        temp_q_df = pd.DataFrame(temp)
        temp_q_df.columns = images_name
        temp_q_df.index = local_methods_name
        q_df.append(temp_q_df)

        q_order_np = np.zeros(temp.shape)
        for image_index in range(images_number):
            temp_list = temp[:, image_index].tolist()
            order_list = sorted(temp_list, reverse=True)
            for method_index in range(methods_number):
                q_order_np[method_index, image_index] = \
                    order_list.index(temp[method_index, image_index]) + 1
        temp_q_order_df = pd.DataFrame(q_order_np)
        temp_q_order_df.columns = images_name
        temp_q_order_df.index = local_methods_name
        q_order_df.append(temp_q_order_df)

    with pd.ExcelWriter(out_excel_address) as writer:
        total_df.to_excel(writer, sheet_name='Total')
        fp_df.to_excel(writer, sheet_name='Total', startrow=methods_number + 3)
        total_order_df.to_excel(writer, sheet_name='Total', startrow=2 * methods_number + 6)
        for metric_index, (q_item, q_order_item) in enumerate(zip(q_df, q_order_df)):
            q_item.insert(0, 'Average', q_item.mean(1))
            q_item.to_excel(writer, sheet_name=local_metrics_name[metric_index])
            q_order_item.insert(0, 'Average', q_order_item.mean(1))
            q_order_item.to_excel(writer, sheet_name=local_metrics_name[metric_index],
                                  startrow=methods_number + 3)


def multi_focus():
    """
    This will expand long duration, please specify methods_id and metrics_id
    """
    cwd = os.getcwd()
    input_dir = os.path.join(cwd, "data", "multi_focus")
    output_dir = os.path.join(cwd, "data", "result", "multi_focus")
    fused_dir = os.path.join(output_dir, "different_methods")
    record_dir = os.path.join(output_dir, "record")
    make_out_dir(fused_dir)
    make_out_dir(record_dir)
    log_address = os.path.join(output_dir, "log.txt")

    # # fuse images by methods in methods_id
    # methods_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
    methods_id = [4]
    fuse_images(input_dir, output_dir, methods_id=methods_id, log_address=log_address)
    #
    # # evaluate result by metrics in metrics_id
    # metrics_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    # methods_list = os.listdir(fused_dir)
    # evaluate_fused(input_dir, fused_dir, record_dir, methods_list, metrics_id=metrics_id, log_address=log_address)

    # export result in excel file
    metrics_id = [3, 4, 11]  # we choose these metrics shown in the paper（Qg, Qm, Qcb）
    records_list = os.listdir(record_dir)
    ablation_list = ['evaluator_se_absmax.npy', 'evaluator_se_average.npy', 'evaluator_se_l1_norm.npy',
                     'evaluator_se_max.npy', 'evaluator_se_sf.npy', 'evaluator_dense_sf_dm.npy',
                     'evaluator_se_sf_dm.npy','evaluator_sse_sf_dm.npy','evaluator_scse_sf_dm.npy']
    
    # Next you should only choose one
    # For ablation experiment
    # records_list = ablation_list

    # For comparison with other methods
    comparison_set = set(records_list) - set(ablation_list)
    records_list = [item for item in comparison_set]
    records_list.append('evaluator_se_sf_dm.npy')

    export_result(input_dir, output_dir, record_dir, records_list, metrics_id)


if __name__ == "__main__":
    multi_focus()
