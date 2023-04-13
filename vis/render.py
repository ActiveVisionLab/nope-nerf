

import os
import sys
import argparse
import time
import torch

sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
from model.checkpoints import CheckpointIO
from model.common import convert3x4_4x4,  interp_poses, interp_poses_bspline, generate_spiral_nerf
from model.extracting_images import Extract_Images
import model as mdl
import imageio
import numpy as np

torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Extract images.'
)
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['extract_images']['extraction_dir'])

# Model
model_cfg = cfg['model']
network_type = cfg['model']['network_type']
if network_type=='official':
    model = mdl.OfficialStaticNerf(cfg)

rendering_cfg = cfg['rendering']
renderer = mdl.Renderer(model, rendering_cfg, device=device)

# init model
nope_nerf = mdl.get_model(renderer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=nope_nerf)
load_dict = checkpoint_io.load(cfg['extract_images']['model_file'])
it = load_dict.get('it', -1)

op = cfg['extract_images']['traj_option']
N_novel_imgs = cfg['extract_images']['N_novel_imgs']

train_loader, train_dataset = get_dataloader(cfg, mode='render', shuffle=False, n_views=N_novel_imgs)
n_views = train_dataset['img'].N_imgs

if cfg['pose']['learn_pose']:
    if cfg['pose']['init_pose']:
        init_pose = train_dataset['img'].c2ws 
    else:
        init_pose = None
    pose_param_net = mdl.LearnPose(n_views, cfg['pose']['learn_R'], cfg['pose']['learn_t'], cfg=cfg, init_c2w=init_pose).to(device=device)
    checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=pose_param_net)
    checkpoint_io_pose.load(cfg['extract_images']['model_file_pose'])
    learned_poses = torch.stack([pose_param_net(i) for i in range(n_views)])
    
    if op=='sprial':
        bds = np.array([2., 4.])
        hwf = train_dataset['img'].hwf
        c2ws = generate_spiral_nerf(learned_poses, bds, N_novel_imgs, hwf)
        c2ws = convert3x4_4x4(c2ws)
    elif op =='interp':
        c2ws = interp_poses(learned_poses.detach().cpu(), N_novel_imgs)
    elif op=='bspline':
        i_train = train_dataset['img'].i_train
        degree=cfg['extract_images']['bspline_degree']
        c2ws = interp_poses_bspline(learned_poses.detach().cpu(), N_novel_imgs, i_train,degree)

c2ws = c2ws.to(device)
if cfg['pose']['learn_focal']:
    focal_net = mdl.LearnFocal(cfg['pose']['learn_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'])
    checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net)
    checkpoint_io_focal.load(cfg['extract_images']['model_file_focal'])
    fxfy = focal_net(0)
    print('learned fx: {0:.2f}, fy: {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
else:
    fxfy = None
# Generator
generator = Extract_Images(
    renderer,cfg,use_learnt_poses=cfg['pose']['learn_pose'],
    use_learnt_focal=cfg['pose']['learn_focal'],
    device=device, render_type=cfg['rendering']['type']
)

# Generate
model.eval()

render_dir = os.path.join(generation_dir, 'extracted_images', op)
if not os.path.exists(render_dir):
    os.makedirs(render_dir)

imgs = []
depths = []
geos = []
output_geo = False
for data in train_loader:
    out = generator.generate_images(data, render_dir, c2ws, fxfy, it, output_geo)
    imgs.append(out['img'])
    depths.append(out['depth'])
    geos.append(out['geo'])
imgs = np.stack(imgs, axis=0)
depths = np.stack(depths, axis=0)

video_out_dir = os.path.join(render_dir, 'video_out')
if not os.path.exists(video_out_dir):
    os.makedirs(video_out_dir)
imageio.mimwrite(os.path.join(video_out_dir, 'img.mp4'), imgs, fps=30, quality=9)
imageio.mimwrite(os.path.join(video_out_dir, 'depth.mp4'), depths, fps=30, quality=9)
if output_geo:  
    geos = np.stack(geos, axis=0)
    imageio.mimwrite(os.path.join(video_out_dir, 'geo.mp4'), geos, fps=30, quality=9)


       
