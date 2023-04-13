import os
import sys
import argparse
import torch
sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
import model as mdl
import numpy as np

from utils_poses.vis_cam_traj import draw_camera_frustum_geometry
from utils_poses.align_traj import align_ate_c2b_use_a2b
from utils_poses.comp_ate import compute_rpe, compute_ATE
import ATE.transformations as tf
torch.manual_seed(0)

# Config
parser = argparse.ArgumentParser(
    description='Eval Poses.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--vis',action='store_true')
args = parser.parse_args()
cfg = load_config(args.config, 'configs/default.yaml')

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")


out_dir = cfg['training']['out_dir']

test_loader, field = get_dataloader(cfg, mode='train', shuffle=False)
N_imgs = field['img'].N_imgs
with torch.no_grad():
    if cfg['pose']['init_pose']:
        if cfg['pose']['init_pose_type']=='gt':
            init_pose = field['img'].c2ws # init with colmap
        elif cfg['pose']['init_pose_type']=='colmap':
            init_pose = field['img'].c2ws_colmap
    else:
        init_pose = None
    pose_param_net = mdl.LearnPose(N_imgs, cfg['pose']['learn_R'], 
                            cfg['pose']['learn_t'], cfg=cfg, init_c2w=init_pose).to(device=device)
    checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=pose_param_net)
    checkpoint_io_pose.load(cfg['extract_images']['model_file_pose'], device)
    learned_poses = torch.stack([pose_param_net(i) for i in range(N_imgs)])

    H = field['img'].H
    W = field['img'].W
    gt_poses = field['img'].c2ws
    if cfg['pose']['learn_focal']:
        focal_net = mdl.LearnFocal(cfg['pose']['learn_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'])
        checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net)
        checkpoint_io_focal.load(cfg['extract_images']['model_file_focal'], device)
        fxfy = focal_net(0)
        fx = fxfy[0] * W / 2
        fy = fxfy[1] * H / 2
    else:
        fx = field['img'].focal
        fy = field['img'].focal


'''Define camera frustums'''
frustum_length = 0.1
est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255

'''Align est traj to colmap traj'''
c2ws_est_to_draw_align2cmp = learned_poses.clone()
ATE_align = True

if ATE_align:  # Align learned poses to colmap poses
    c2ws_est_aligned = align_ate_c2b_use_a2b(learned_poses, gt_poses)  # (N, 4, 4)
    c2ws_est_to_draw_align2cmp = c2ws_est_aligned

    # compute ate
    ate = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    print("{0:.3f}".format(rpe_trans*100),'&' "{0:.3f}".format(rpe_rot * 180 / np.pi), '&', "{0:.3f}".format(ate))


if args.vis:
    import open3d as o3d
    frustum_est_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                    fx, fy,
                                                    frustum_length, est_traj_color)
    frustum_colmap_list = draw_camera_frustum_geometry(gt_poses.cpu().numpy(), H, W,
                                                        fx, fy,
                                                        frustum_length, cmp_traj_color)

    geometry_to_draw = []
    geometry_to_draw.append(frustum_est_list)
    geometry_to_draw.append(frustum_colmap_list)

    '''o3d for line drawing'''
    t_est_list = c2ws_est_to_draw_align2cmp[:, :3, 3]
    t_cmp_list = gt_poses[:, :3, 3]

    '''line set to note pose correspondence between two trajs'''
    line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
    line_ends = [[i, i+N_imgs] for i in range(N_imgs)]  # (N, 2) connect two end points.


    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)
    unit_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=2)
    unit_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(unit_sphere)
    unit_sphere.paint_uniform_color((0, 1, 0))
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()

    geometry_to_draw.append(line_set)

    o3d.visualization.draw_geometries(geometry_to_draw)





