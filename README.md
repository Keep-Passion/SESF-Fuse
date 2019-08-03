# SESF-Fuse
SESF-Fuse: An unsupervised deep model used for multi-focus image fusion

## Branches Introduction
We provide a general image fusion framework in this branch, which include all the experiments in our paper. Besides, one can easily modify it for new experiment.  
We provide the training and testing method of SESF-Fuse in [master branch](https://github.com/MATony/SESF-Fuse/tree/master).

## Content
We provide a general framework which consists of almost image fusion algorithms (including our SESF-Fuse) and metrics. All the codes which copied from other site are cited in the script.  
* Algorithms:    
gf, dsift, focus_stack, sf, nsct, cvt, dwt, lp, rp, dtcwt, sr, mwg, imf, cnn_fuse, dense_fuse_1e3_add, dense_fuse_1e3_l1, deep_fuse, fusion_gan  
* Metrics:   
Qmi, Qte, Qncie, Qg, Qm, Qsf, Qp, Qs, Qc, Qy, Qcv, Qcb, VIFF, MEF_SSIM, SSIM_A, FMI_EDGE, FMI_DCT, FMI_W, Nabf, SCD, SD, SF, CC  

## Requirements
Python 3.6, matlab 2018Rb, Pytorch 1.1, Tensorflow 1.13.

## Citation
Still in submission.
