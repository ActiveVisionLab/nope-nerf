# NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior

**[Project Page](https://nope-nerf.active.vision/) | [Arxiv](https://arxiv.org/abs/2212.07388) | [Data](https://www.robots.ox.ac.uk/~wenjing/Tanks.zip) | [Pretrained Model](https://www.robots.ox.ac.uk/~wenjing/pretrained_Tanks.zip)**

Wenjing Bian, 
Zirui Wang, 
[Kejie Li](https://likojack.github.io/kejieli/#/home), 
[Jiawag Bian](https://jwbian.net/),
[Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor/). (CVPR 2023 highlight)

Active Vision Lab, University of Oxford.


## Table of Content
- [Installation](#Installation)
- [Data](#Data)
- [Usage](#Usage)
- [Acknowledgement](#Acknowledgement)
- [Citation](#citation)

## Installation

```
git clone https://github.com/ActiveVisionLab/nope-nerf.git
cd nope-nerf
conda env create -f environment.yaml
conda activate nope-nerf
```

## Data and Preprocessing
1. [Tanks and Temples](https://www.robots.ox.ac.uk/~wenjing/Tanks.zip):
Our pre-processed Tanks and Temples data contains the 8 scenes shown in the paper. Each scene contains images, monocular depth estimations from DPT and COLMAP poses. You can download and unzip it to `data` directory.

2. [NeRF LLFF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1):
We also provide config file for NeRF LLFF dataset. You can download the dataset and unzip it to `data` directory. One example of the config file is `configs/LLFF/fern.yaml`. 


3. If you want to use your own image sequence with customised camera intrinsics, you need to add an `intrinsics.npz` file to the scene directory. One example of the config file is `configs/Test/images.yaml` (please add your own data to the `data/Test/images` directory). 



Monocular depth map generation: you can first download the pre-trained DPT model from [this link](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing) provided by [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT) to `DPT` directory, then run
```
python preprocess/dpt_depth.py configs/preprocess.yaml
```
to generate monocular depth maps. You need to modify the `cfg['dataloading']['path']` and `cfg['dataloading']['scene']` in `configs/preprocess.yaml` to your own image sequence.

## Training

1. Train a new model from scratch:

```
python train.py configs/Tanks/Ignatius.yaml
```
where you can replace `configs/Tanks/Ignatius.yaml` with other config files.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
tensorboard --logdir ./out --port 6006
```

For available training options, please take a look at `configs/default.yaml`.
## Evaluation
1. Evaluate image quality and depth:
```
python evaluation/eval.py configs/Tanks/Ignatius.yaml
```
To evaluate depth: add `--depth` . Note that you need to add ground truth depth maps by yourself.

2. Evaluate poses:
```
python evaluation/eval_poses.py configs/Tanks/Ignatius.yaml
```
To visualise estimated & ground truth trajectories: add `--vis` 


## More Visualisations
Novel view synthesis
```
python vis/render.py configs/Tanks/Ignatius.yaml
```
Pose visualisation (estimated trajectory only)
```
python vis/vis_poses.py configs/Tanks/Ignatius.yaml
```


## Acknowledgement
We thank [Theo Costain](https://www.robots.ox.ac.uk/~costain/) and Michael Hobley for helpful comments and proofreading. We thank Shuai Chen and Xinghui Li for insightful discussions. Wenjing Bian is supported by China Scholarship Council (CSC).
 
We refer to [NeRFmm](https://github.com/ActiveVisionLab/nerfmm), [UNISURF](https://github.com/autonomousvision/unisurf), [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT), [kitti-odom-eval](https://github.com/Huangying-Zhan/kitti-odom-eval) and [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). We thank the excellent code they provide.

## Citation
```
 @inproceedings{bian2022nopenerf,
	author    = {Wenjing Bian and Zirui Wang and Kejie Li and Jiawang Bian and Victor Adrian Prisacariu},
	title     = {NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior},
	journal   = {CVPR},
	year      = {2023}
	}
```