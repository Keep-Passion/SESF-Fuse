# SESF-Fuse
SESF-Fuse: An Unsupervised Deep Model for Multi-Focus Image Fusion

## Abstract
In this work, we propose an unsupervised deep learning model to address multi-focus image fusion problem. First, we train an encoder-decoder architecture in unsupervised manner to acquire deep feature of input images. And then we utilized those features and spatial frequency to measure activity level, which plays crucial role in multi-focus fusion task. The key point behind of proposed method is that only the objects within the depth-of-field (DOF) have sharp appearance in the photograph while other objects are likely to be blurred. In contrast to previous works, our method analysis sharp appearance in deep feature instead of original image. Experimental results demonstrate that the proposed method achieve the state-of-art fusion performance compared to existing 16 fusion methods in objective and subjective assessment. 

## Visualization
We show the visualization of fused result in next figure. The first row is near focused source image and the second row is far focused source image. The third row is decision map of our method and the final row is fused result.
![image](https://github.com/Keep-Passion/SESF-Fuse/blob/master/nets/figure/visualization.png)

## Branches Introduction
We provide the training and testing method of SESF-Fuse in this branch.  
We provide a general image fusion framework in [experiment branch](https://github.com/Keep-Passion/SESF-Fuse/tree/Experiment), which include all the experiments in our paper. Besides, one can easily modify it for new experiment.

## Requirements
Pytorch, Python3.6

## Citation
If you use it successfully for your research please be so kind to cite [the paper](https://link.springer.com/article/10.1007/s00521-020-05358-9#citeas).

Ma, B., Zhu, Y., Yin, X. et al. SESF-Fuse: an unsupervised deep model for multi-focus image fusion. Neural Comput & Applic (2020). https://doi.org/10.1007/s00521-020-05358-9

## Acknowledgement
The authors acknowledge the financial support from the National Key Research and Development Program of China (No. 2016YFB0700500).

## Recommendation
Our new work GACN can be found at [the paper](https://arxiv.org/abs/2010.08751) and [the code](https://github.com/Keep-Passion/GACN).
