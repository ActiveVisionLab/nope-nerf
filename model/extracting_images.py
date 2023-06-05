import os
import torch
from collections import defaultdict
from model.common import (
    get_tensor_values, arange_pixels
)
from tqdm import tqdm
import logging
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image
import imageio
class Extract_Images(object):
    def __init__(self, renderer, cfg, use_learnt_poses=True, use_learnt_focal=True, device=None,render_type=None):
        self.points_batch_size = 100000
        self.renderer = renderer
        self.resolution = cfg['extract_images']['resolution']
        self.device = device
        self.use_learnt_poses = use_learnt_poses
        self.use_learnt_focal = use_learnt_focal
        self.render_type = render_type

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        img_idx = data.get('img.idx')
        # world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)

        return (camera_mat, scale_mat, img_idx)

    def generate_images(self, data, render_dir, c2ws, fxfy, it, output_geo):
        self.renderer.eval()
        (camera_mat, scale_mat, img_idx) = self.process_data_dict(data)
        img_idx = int(img_idx)
        if self.use_learnt_poses:
            c2w = c2ws[img_idx]
            world_mat = torch.inverse(c2w).unsqueeze(0)
        if self.use_learnt_focal:
            camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0], 
                    [0, -fxfy[1], 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]]]).to(self.device)
        h, w = self.resolution
        
        p_loc, pixels = arange_pixels(resolution=(h, w))

        pixels = pixels.to(self.device)
        

        # redundancy, set depth_input values to ones to avoid masking
        depth_input = torch.zeros(1, 1, h, w).to(self.device)
        depth_input = get_tensor_values(depth_input, pixels.clone(), mode='nearest', scale=True, detach=False)
        depth_input = torch.ones_like(depth_input)
        with torch.no_grad():
            rgb_pred = []
            depth_pred = []
            for ii, (pixels_i, depth_i) in enumerate(zip(torch.split(pixels, self.points_batch_size, dim=1), torch.split(depth_input, self.points_batch_size, dim=1))):
                out_dict = self.renderer(pixels_i, depth_i, camera_mat, world_mat, scale_mat, 
                self.render_type, eval_=True, it=it, add_noise=False)
                rgb_pred_i = out_dict['rgb']
                rgb_pred.append(rgb_pred_i)
                depth_pred_i = out_dict['depth_pred']
                depth_pred.append(depth_pred_i)
            rgb_pred = torch.cat(rgb_pred, dim=1)
            rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
            depth_pred = torch.cat(depth_pred, dim=0)
            depth_pred = depth_pred.view(h, w).detach().cpu().numpy()

            img_out = (rgb_pred * 255).astype(np.uint8)
            depth_out = depth_pred


        if output_geo:
            with torch.no_grad():
                mask_pred = torch.ones(pixels.shape[0], pixels.shape[1]).bool()
                rgb_pred = \
                    [self.renderer(
                        pixels_i, None, camera_mat, world_mat, scale_mat, 
                    'phong_renderer', eval_=True, it=it, add_noise=False)['rgb']
                        for ii, pixels_i in enumerate(torch.split(pixels, 1024, dim=1))]
            
                rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
                p_loc1 = p_loc[mask_pred]
                geo_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)

                if mask_pred.sum() > 0:
                    rgb_hat = rgb_pred[mask_pred].detach().cpu().numpy()
                    rgb_hat = (rgb_hat * 255).astype(np.uint8)
                    geo_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
            geo_out_dir = os.path.join(render_dir, 'geo_out')
            if not os.path.exists(geo_out_dir):
                os.makedirs(geo_out_dir)
            imageio.imwrite(os.path.join(geo_out_dir, str(img_idx).zfill(4) + '.png'), geo_out)
        else:
            geo_out = None

        img_out_dir = os.path.join(render_dir, 'img_out')
        depth_out_dir = os.path.join(render_dir, 'depth_out')
        
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)
        if not os.path.exists(depth_out_dir):
            os.makedirs(depth_out_dir)

        filename = os.path.join(depth_out_dir, '{}.npy'.format(img_idx))
        np.save(filename, depth_out)
        
        depth_out = (np.clip(255.0 / depth_out.max() * (depth_out - depth_out.min()), 0, 255)).astype(np.uint8)
        
        imageio.imwrite(os.path.join(img_out_dir, str(img_idx).zfill(4) + '.png'), img_out)
        imageio.imwrite(os.path.join(depth_out_dir, str(img_idx).zfill(4) + '.png'), depth_out)
        

        img_dict = {'img': img_out,
                    'depth': depth_out,
                    'geo': geo_out}
        return img_dict