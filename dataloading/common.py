import os
import torch
from PIL import Image
import numpy as np
import imageio
import cv2

def _minify(basedir, factors=[], resolutions=[], img_folder='images'):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, img_folder + '_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, img_folder + '_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, img_folder)
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = img_folder + '_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = img_folder + '_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, crop_size=0, load_colmap_poses=True):
    if load_colmap_poses:
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
        bds = poses_arr[:, -2:].transpose([1,0])
    img_folder = 'images'
    crop_ratio = 1
    focal_crop_factor = 1
    if crop_size!=0:
        img_folder = 'images_cropped'
        crop_dir = os.path.join(basedir, 'images_cropped')
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
        for f in sorted(os.listdir(os.path.join(basedir, 'images'))):
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
                image = imageio.imread(os.path.join(basedir, 'images', f))
                crop_size_H = crop_size
                H, W, _ = image.shape
                crop_size_W = int(crop_size_H * W/H)
                image_cropped = image[crop_size_H:H-crop_size_H, crop_size_W:W-crop_size_W]
                save_path = os.path.join(crop_dir, f)
                im = Image.fromarray(image_cropped)
                im = im.resize((W, H))
                im.save(save_path)
        crop_ratio = crop_size_H / H
        print('=======images cropped=======')
        focal_crop_factor = (H - 2*crop_size_H) / H
            


    
    img0 = [os.path.join(basedir, img_folder, f) for f in sorted(os.listdir(os.path.join(basedir, img_folder))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor], img_folder=img_folder)
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]], img_folder=img_folder)
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]], img_folder=img_folder)
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, img_folder + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    sh = imageio.imread(imgfiles[0]).shape
    if load_colmap_poses:
        if poses.shape[-1] != len(imgfiles):
            print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
            return
    
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1./factor 
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  

    if load_colmap_poses:
        print('Loaded image data', imgs.shape, poses[:,-1,0])
    else:
        print('Loaded image data', imgs.shape)
        poses=None
        bds=None
    # added
    imgnames = [f for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    return poses, bds, imgs, imgnames, crop_ratio, focal_crop_factor
def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses
def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m
def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds


def load_gt_depths(image_list, datadir, H=None, W=None, crop_ratio=1):
    depths = []
    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, 'depth', '{}.png'.format(frame_id))
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000
        if crop_ratio != 1:
            h, w = depth.shape
            crop_size_h = int(h*crop_ratio)
            crop_size_w = int(w*crop_ratio)
            depth = depth[crop_size_h:h-crop_size_h, crop_size_w:w-crop_size_w]
        
        if H is not None:
            # mask = (depth > 0).astype(np.uint8)
            depth_resize = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            # mask_resize = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            depths.append(depth_resize)
            # masks.append(mask_resize > 0.5)
        else:
            depths.append(depth)
            # masks.append(depth > 0)
    return np.stack(depths)
def load_depths(image_list, datadir, H=None, W=None):
    depths = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, '{}_depth.npy'.format(frame_id))
        if not os.path.exists(depth_path):
            depth_path = os.path.join(datadir, 'depth_{}.npy'.format(frame_id))
        depth = np.load(depth_path)
        
        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)
    return np.stack(depths)
def load_images(image_list, datadir):
    images = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        im_path = os.path.join(datadir, '{}.npy'.format(frame_id))
        im = np.load(im_path)
        images.append(im)
    return np.stack(images)
def load_depths_npz(image_list, datadir, H=None, W=None, norm=False):
    depths = []

    for image_name in image_list:
        frame_id = image_name.split('.')[0]
        depth_path = os.path.join(datadir, 'depth_{}.npz'.format(frame_id))
        depth = np.load(depth_path)['pred']
        if depth.shape[0] == 1:
            depth = depth[0]

        if H is not None:
            depth_resize = cv2.resize(depth, (W, H))
            depths.append(depth_resize)
        else:
            depths.append(depth)
    depths = np.stack(depths)
    if norm:
        depths_n = []
        t_all = np.median(depths)
        s_all = np.mean(np.abs(depths - t_all))
        for depth in depths:
            t_i = np.median(depth)
            s_i = np.mean(np.abs(depth - t_i))
            depth = s_all * (depth - t_i) / s_i + t_all
            depths_n.append(depth)
        depths = np.stack(depths_n)
    return depths