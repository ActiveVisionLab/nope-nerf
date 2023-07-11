import os
import glob
import random
import logging
import torch
from PIL import Image
import numpy as np
import imageio
import cv2
from dataloading.common import _load_data, recenter_poses, spherify_poses, load_depths_npz, load_gt_depths
logger = logging.getLogger(__name__)

class DataField(object):
    def __init__(self, model_path,
                 transform=None, 
                 with_camera=False, 
                with_depth=False,
                 use_DPT=False, scene_name=[' '], mode='train', spherify=False, 
                 load_ref_img=False,customized_poses=False,
                 customized_focal=False,resize_factor=2, depth_net='dpt',crop_size=0, 
                 random_ref=False,norm_depth=False,load_colmap_poses=True, sample_rate=8, **kwargs):
        """load images, depth maps, etc.
        Args:
            model_path (str): path of dataset
            transform (class, optional):  transform made to the image. Defaults to None.
            with_camera (bool, optional): load camera intrinsics. Defaults to False.
            with_depth (bool, optional): load gt depth maps (if available). Defaults to False.
            DPT (bool, optional): run DPT model. Defaults to False.
            scene_name (list, optional): scene folder name. Defaults to [' '].
            mode (str, optional): train/eval/all/render. Defaults to 'train'.
            spherify (bool, optional): spherify colmap poses (no effect to training). Defaults to False.
            load_ref_img (bool, optional): load reference image. Defaults to False.
            customized_poses (bool, optional): use GT pose if available. Defaults to False.
            customized_focal (bool, optional): use GT focal if provided. Defaults to False.
            resize_factor (int, optional): image downsample factor. Defaults to 2.
            depth_net (str, optional): which depth estimator use. Defaults to 'dpt'.
            crop_size (int, optional): crop if images have black border. Defaults to 0.
            random_ref (bool/int, optional): if use a random reference image/number of neaest images. Defaults to False.
            norm_depth (bool, optional): normalise depth maps. Defaults to False.
            load_colmap_poses (bool, optional): load colmap poses. Defaults to True.
            sample_rate (int, optional): 1 in 'sample_rate' images as test set. Defaults to 8.
        """
        self.transform = transform
        self.with_camera = with_camera
        self.with_depth = with_depth
        self.use_DPT = use_DPT
        self.mode = mode
        self.ref_img = load_ref_img
        self.random_ref = random_ref
        self.sample_rate = sample_rate
        
        load_dir = os.path.join(model_path, scene_name[0])
        if crop_size!=0:
            depth_net = depth_net + '_' + str(crop_size)
        poses, bds, imgs, img_names, crop_ratio, focal_crop_factor = _load_data(load_dir, factor=resize_factor, crop_size=crop_size, load_colmap_poses=load_colmap_poses)
        if load_colmap_poses:
            poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
            poses = np.moveaxis(poses, -1, 0).astype(np.float32)
            bds = np.moveaxis(bds, -1, 0).astype(np.float32)
            bd_factor = 0.75
            # Rescale if bd_factor is provided
            sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
            poses[:,:3,3] *= sc
            bds *= sc
            poses = recenter_poses(poses)
            if spherify:
                poses, render_poses, bds = spherify_poses(poses, bds)
            input_poses = poses.astype(np.float32)
            hwf = input_poses[0,:3,-1]
            self.hwf = input_poses[:,:3,:]
            input_poses = input_poses[:,:3,:4]
            H, W, focal = hwf
            H, W = int(H), int(W)
            poses_tensor = torch.from_numpy(input_poses)
            bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
            bottom = bottom.repeat(poses_tensor.shape[0], 1, 1)
            c2ws_colmap = torch.cat([poses_tensor, bottom], 1)
            

        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        _, _, h, w = imgs.shape

        if customized_focal:
            focal_gt = np.load(os.path.join(load_dir, 'intrinsics.npz'))['K'].astype(np.float32)
            if resize_factor is None:
                resize_factor = 1
            fx = focal_gt[0, 0] / resize_factor
            fy = focal_gt[1, 1] / resize_factor
        else:
            if load_colmap_poses:
                fx, fy = focal, focal
            else:
                print('No focal provided, use image size as default')
                fx, fy = w, h
        fx = fx / focal_crop_factor
        fy = fy / focal_crop_factor
        
        
        self.H, self.W, self.focal = h, w, fx
        self.K = np.array([[2*fx/w, 0, 0, 0], 
            [0, -2*fy/h, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]).astype(np.float32)
        ids = np.arange(imgs.shape[0])
        i_test = ids[int(sample_rate/2)::sample_rate]
        i_train = np.array([i for i in ids if i not in i_test])
        self.i_train = i_train
        self.i_test = i_test
        image_list_train = [img_names[i] for i in i_train]
        image_list_test = [img_names[i] for i in i_test]
        print('test set: ', image_list_test)

        if customized_poses:
            c2ws_gt = np.load(os.path.join(load_dir, 'gt_poses.npz'))['poses'].astype(np.float32)
            T = torch.tensor(np.array([[1, 0, 0, 0],[0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)) # ScanNet coordinate
            c2ws_gt = torch.from_numpy(c2ws_gt)
            c2ws = c2ws_gt @ T
        else:
            if load_colmap_poses:
                c2ws = c2ws_colmap
            else:
                c2ws = None
        
        
        self.N_imgs_train = len(i_train)
        self.N_imgs_test = len(i_test)
        
        pred_depth_path = os.path.join(load_dir, depth_net)
        self.dpt_depth = None
        if mode in ('train','eval_trained', 'render'):
            idx_list = i_train
            self.img_list = image_list_train
        elif mode=='eval':
            idx_list = i_test
            self.img_list = image_list_test
        elif mode=='all':
            idx_list = ids
            self.img_list = img_names

        self.imgs = imgs[idx_list]
        self.N_imgs = len(idx_list)
        if c2ws is not None:
            self.c2ws = c2ws[idx_list]
        if load_colmap_poses:
            self.c2ws_colmap = c2ws_colmap[i_train]
        if not use_DPT:
            self.dpt_depth = load_depths_npz(image_list_train, pred_depth_path, norm=norm_depth)
        if with_depth:
            self.depth = load_gt_depths(image_list_train, load_dir, crop_ratio=crop_ratio)
        

       

    def load(self, input_idx_img=None):
        ''' Loads the field.
        '''
        return self.load_field(input_idx_img)

    def load_image(self, idx, data={}):
        image = self.imgs[idx]
        data[None] = image
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            data['normalised_img'] = data_in['image']
        data['idx'] = idx
    def load_ref_img(self, idx, data={}):
        if self.random_ref:
            if idx==self.N_imgs-1:
                ref_idx = idx-1
            else:
                ran_idx = random.randint(1, min(self.random_ref, self.N_imgs-idx-1))
                ref_idx = idx + ran_idx
        image = self.imgs[ref_idx]
        if self.dpt_depth is not None:
            dpt = self.dpt_depth[ref_idx]
            data['ref_dpts'] = dpt
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            normalised_ref_img = data_in['image']
            data['normalised_ref_img'] = normalised_ref_img
        if self.with_depth:
            depth = self.depth[ref_idx]
            data['ref_depths'] = depth
        data['ref_imgs'] = image
        data['ref_idxs'] = ref_idx

    def load_depth(self, idx, data={}):
        depth = self.depth[idx]
        data['depth'] = depth
    def load_DPT_depth(self, idx, data={}):
        depth_dpt = self.dpt_depth[idx]
        data['dpt'] = depth_dpt

    def load_camera(self, idx, data={}):
        data['camera_mat'] = self.K
        data['scale_mat'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]).astype(np.float32)
        data['idx'] = idx
    
   
        
    def load_field(self, input_idx_img=None):
        if input_idx_img is not None:
            idx_img = input_idx_img
        else:
            idx_img = 0
        # Load the data
        data = {}
        if not self.mode =='render':
            self.load_image(idx_img, data)
            if self.ref_img:
                self.load_ref_img(idx_img, data)
            if self.with_depth:
                self.load_depth(idx_img, data)
            if self.dpt_depth is not None:
                self.load_DPT_depth(idx_img, data)
        if self.with_camera:
            self.load_camera(idx_img, data)
        
        return data





