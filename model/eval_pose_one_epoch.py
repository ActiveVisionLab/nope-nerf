import os
import torch
from model.common import (
    arange_pixels
)
import logging
from model.losses import Loss_Eval
logger_py = logging.getLogger(__name__)

class Trainer_pose(object):
    
    def __init__(self, model, cfg, device=None, optimizer_pose=None, pose_param_net=None, 
                    focal_net=None, **kwargs):
        self.model = model
        self.device = device
        self.optimizer_pose = optimizer_pose
        self.pose_param_net = pose_param_net
        self.focal_net = focal_net
        self.n_points = cfg['n_points']
        self.rendering_technique = cfg['type']

        self.loss = Loss_Eval()


    def train_step(self, data, it=100000):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.model.eval()
        self.pose_param_net.train()
        self.optimizer_pose.zero_grad()
        if self.focal_net is not None:
            self.focal_net.eval()
        loss_dict = self.compute_loss(data, it=it)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer_pose.step()
        return loss_dict


    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
       
        img = data.get('img').to(device)
        img_idx = data.get('img.idx').to(device)
        batch_size, _, h, w = img.shape
        depth_img = data.get('img.depth', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)  # add for nope_nerf
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        return (img, depth_img, camera_mat, scale_mat, img_idx)

    def compute_loss(self, data, eval_mode=False, it=100000):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        n_points = self.n_points
        (img, depth_img, camera_mat, scale_mat, img_idx) = self.process_data_dict(data)
        # Shortcuts
        device = self.device
        batch_size, _, h, w = img.shape
        c2w = self.pose_param_net(img_idx)
        world_mat = torch.inverse(c2w).unsqueeze(0)
        if self.focal_net is not None:
            fxfy = self.focal_net(0)
            pad = torch.zeros(4)
            one = torch.tensor([1])
            camera_mat = torch.cat([fxfy[0:1], pad, -fxfy[1:2], pad, -one, pad, one]).to(device)
            camera_mat = camera_mat.view(1, 4, 4)
        
        
        ray_idx = torch.randperm(h*w,device=device)[:n_points]
        img_flat = img.view(batch_size, 3, h*w).permute(0,2,1)
        rgb_gt = img_flat[:,ray_idx]
        p_full = arange_pixels((h, w), batch_size)[1].to(device)
        p = p_full[:, ray_idx]
        pix = ray_idx
        
        out_dict = self.model(
            p, pix, camera_mat, world_mat, scale_mat, 
            self.rendering_technique, it=it,
            eval_mode=True, depth_img=depth_img,  add_noise=False,img_size=(h, w)
        )
        loss_dict = self.loss(out_dict['rgb'], rgb_gt)
        return loss_dict
