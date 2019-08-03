# SESF-Fuse
SESF-Fuse: An unsupervised deep model used for multi-focus image fusion

## Abstract
In this work, we propose an unsupervised deep learning model to address multi-focus image fusion problem. First, we train an encoder-decoder architecture in unsupervised manner to acquire deep feature of input images. And then we utilized those features and spatial frequency to measure activity level, which plays crucial role in multi-focus fusion task. The key point behind of proposed method is that only the objects within the depth-of-field (DOF) have sharp appearance in the photograph while other objects are likely to be blurred. In contrast to previous works, our method analysis sharp appearance in deep feature instead of original image. Experimental results demonstrate that the proposed method achieve the state-of-art fusion performance compared to existing 18 fusion methods in objective and subjective assessment. 

## Visualization
We show the visualization of fused result in next figure. The first row is near focused source image and the second row is far focused source image. The third row is decision map of our method and the final row is fused result.
![image](https://github.com/MATony/SESF-Fuse/blob/master/nets/figure/visualization.png)

## Branches Introduction
We provide the training and testing method of SESF-Fuse in this branch.  
We provide a general image fusion framework in [experiment branch](https://github.com/MATony/SESF-Fuse/tree/Experiment), which include all the experiments in our paper. Besides, one can easily modify it for new experiment.

## Requirements
Pytorch, Python3.6

## Citation
Still in submission.
