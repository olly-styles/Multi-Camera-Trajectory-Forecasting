# Multi-Camera-Trajectory-Forecasting

This repo contains information on the Warwick-NTU Multi-camera Forecasting database (WNMF) and baseline multi-camera trajectory forecasting (MCTF) experiments. This repo acompanies the following paper:

Olly Styles, Tanaya Guha, Victor Sanchez, Alex C. Kot, “Multi-Camera Trajectory Forecasting: Pedestrian Trajectory Prediction in a Network of Cameras”, IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2020

Paper link: https://arxiv.org/abs/2005.00282

<p align="center"> 
<img src="mctf.jpg" width=50%>
</p>

## Changelog

2020.05.01 - Initial dataset release

2021.08.11 - Cleaned annotations, preprocessing new MCTF problems, more MCTF models, multi-target MCTF preprocessing. Updated annotations are avaiable upon dataset request, and new code is available in the our new repositry, [[Trajectory Tensors](https://github.com/olly-styles/Trajectory-Tensors)]. 

2024.04.13 - Some files in WNMF have become corrupt - these are `WNMF_videos/day_20/departures/departure_027.mp4` and `WNMF\data\reid_features\departure_features_day_3.npy`



## Accessing WNMF
If you are interested in downloading the WNMF dataset, please see [[this page](https://rose1.ntu.edu.sg/dataset/Warwick-NTU/)] to request access. 

## Dataset details

The data download contains the following:

##### Videos
Videos are paired into entrances and departures. A departure is defined as the 4 seconds before tracking infromation is lost (and the person is therefore assumed to have left the camera view. An entrance is the next camera of re-apperance for this individual. Entrance video clips last for 12 seconds, starting from the moment the individual departed from the other camera view. Each video is processed using [[RetinaFace](https://arxiv.org/abs/1905.00641)] using an open-source [[Pytorch implementation](https://github.com/biubug6/Pytorch_Retinaface)] to mask faces.
##### Bounding boxes
Bouding boxes are obtained using an [[open-source implementation](https://github.com/matterport/Mask_RCNN)] of [[Mask-RCNN](https://arxiv.org/abs/1703.06870)], pre-trained on [[MS-COCO](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)]. Individuals are then tracked using an [[open-source implementation](https://github.com/Qidian213/deep_sort_yolov3)] of the [[DeepSORT](https://arxiv.org/abs/1703.07402)] tracking algorithm.
##### Entrances and departures
Each track is labelled as as entrance (first frame of the track) or departure (last frame of the track)
##### RE-ID features
RE-ID features are computed using an [[open-source implementation](https://github.com/michuanhaohao/reid-strong-baseline)] of the [[bag-of-tricks](https://arxiv.org/abs/1903.07071)] RE-ID model pretrained on [[Market-1501](https://ieeexplore.ieee.org/document/7410490/)].
##### Cross-camera matches
Cross-camera matches are found using the labelling procedure described in our paper.
##### Models
Pre-trained weights for each model in our paper.
##### Camera topology
The camera topology is shown in the figure below.
<p align="center"> 
  <img src="camera_layout_and_frames.jpg" width=50%>
</p>
