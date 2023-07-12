import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Loss_Eval(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, rgb_pred, rgb_gt):
        loss = F.mse_loss(rgb_pred, rgb_gt)
        return_dict = {
            'loss': loss
        }
        return return_dict
    
class Loss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        
        self.depth_loss_type = cfg['depth_loss_type']

        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')

        self.cfg = cfg

    def get_rgb_full_loss(self, rgb_values, rgb_gt, rgb_loss_type='l2'):
        if rgb_loss_type == 'l1':
            rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])
        elif rgb_loss_type == 'l2':
            rgb_loss = self.l2_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])            
        return rgb_loss

    def depth_loss_dpt(self, pred_depth, gt_depth, weight=None):
        """
        :param pred_depth:  (H, W)
        :param gt_depth:    (H, W)
        :param weight:      (H, W)
        :return:            scalar
        """
        
        t_pred = torch.median(pred_depth)
        s_pred = torch.mean(torch.abs(pred_depth - t_pred))

        t_gt = torch.median(gt_depth)
        s_gt = torch.mean(torch.abs(gt_depth - t_gt))

        pred_depth_n = (pred_depth - t_pred) / s_pred
        gt_depth_n = (gt_depth - t_gt) / s_gt

        if weight is not None:
            loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
            loss = loss * weight
            loss = loss.sum() / (weight.sum() + 1e-8)
        else:
            loss = F.mse_loss(pred_depth_n, gt_depth_n)
        return loss
    
    def get_depth_loss(self, depth_pred, depth_gt):
        if self.depth_loss_type == 'l1':
            loss = self.l1_loss(depth_pred, depth_gt) / float(depth_pred.shape[0])
        elif self.depth_loss_type=='invariant':
            loss = self.depth_loss_dpt(depth_pred, depth_gt)
        return loss
    def get_reprojection_loss(self, rgb, rgb_refs, valid_points, rgb_refs_ori):
        cfg = self.cfg
        loss = 0
        for (rgb_ref, rgb_ref_ori) in zip(rgb_refs, rgb_refs_ori):
            diff_img = (rgb - rgb_ref).abs()
            if cfg['with_auto_mask'] == True:
                auto_mask = (diff_img.mean(dim=-1, keepdim=True) < (rgb - rgb_ref_ori).abs().mean(dim=-1, keepdim=True)).float() * valid_points
                valid_points = auto_mask
            loss = loss + self.mean_on_mask(diff_img, valid_points)
        loss = loss / len(rgb_refs)
        return loss
    # compute mean value given a binary mask
    def mean_on_mask(self, diff, valid_mask):
        mask = valid_mask.expand_as(diff)
        if mask.sum() > 0:
            mean_value = (diff[mask]).sum() / mask.sum()
            # mean_value = (diff * mask).sum() / mask.sum()
        else:
            print('============invalid mask==========')
            mean_value = torch.tensor(0).float().cuda()
        return mean_value
    def get_DPT_reprojection_loss(self, rgb, rgb_refs, valid_points, rgb_img_refs_ori):
        cfg = self.cfg
        loss = 0
        for rgb_ref, rgb_img_ref_ori in zip(rgb_refs, rgb_img_refs_ori):
            diff_img = (rgb - rgb_ref).abs()
            diff_img = diff_img.clamp(0, 1)
            if cfg['with_auto_mask'] == True:
                auto_mask = (diff_img.mean(dim=1, keepdim=True) < (rgb - rgb_img_ref_ori).abs().mean(dim=1, keepdim=True)).float() 
                auto_mask = auto_mask* valid_points
                valid_points = auto_mask

            if cfg['with_ssim'] == True:
                ssim_map = compute_ssim_loss(rgb, rgb_ref)
                diff_img = (0.15 * diff_img + 0.85 * ssim_map)
            loss = loss + self.mean_on_mask(diff_img, valid_points)
        loss = loss / len(rgb_refs)
        return loss
    def get_weight_dist_loss(self, t_list):
        dist = t_list - t_list.roll(shifts=1, dims=0)
        dist = dist[1:]  # the first dist is meaningless
        dist = dist.norm(dim=1)  # (N-1, )
        dist_diff = dist - dist.roll(shifts=1)
        dist_diff = dist_diff[1:]  # (N-2, )

        loss_dist_1st = dist.mean()
        loss_dist_2nd = dist_diff.pow(2.0).mean()
        return loss_dist_1st, loss_dist_2nd
    
    def get_pc_loss(self, Xt, Yt):
        # compute  error
        match_method = self.cfg['match_method']
        if match_method=='dense':
            loss1 = self.comp_point_point_error(Xt[0].permute(1, 0), Yt[0].permute(1, 0))
            loss2= self.comp_point_point_error(Yt[0].permute(1, 0), Xt[0].permute(1, 0))
            loss = loss1 + loss2
        return loss
    def get_depth_consistency_loss(self, d1_proj, d2, d2_proj=None, d1=None):
        loss = self.l1_loss(d1_proj, d2) / float(d1_proj.shape[1])
        if d2_proj is not None:
            loss = 0.5 * loss + 0.5 * self.l1_loss(d2_proj, d1) / float(d2_proj.shape[1])
        return loss
    def comp_closest_pts_idx_with_split(self, pts_src, pts_des):
        """
        :param pts_src:     (3, S)
        :param pts_des:     (3, D)
        :param num_split:
        :return:
        """
        pts_src_list = torch.split(pts_src, 500000, dim=1)
        idx_list = []
        for pts_src_sec in pts_src_list:
            diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]  # (3, S, 1) - (3, 1, D) -> (3, S, D)
            dist = torch.linalg.norm(diff, dim=0)  # (S, D)
            closest_idx = torch.argmin(dist, dim=1)  # (S,)
            idx_list.append(closest_idx)
        closest_idx = torch.cat(idx_list)
        return closest_idx
    def comp_point_point_error(self, Xt, Yt):
        closest_idx = self.comp_closest_pts_idx_with_split(Xt, Yt)
        pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
        pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
        eng = torch.mean(pt_pt_dist)
        return eng
   
    def get_rgb_s_loss(self, rgb1, rgb2, valid_points):
        diff_img = (rgb1 - rgb2).abs()
        diff_img = diff_img.clamp(0, 1)
        if self.cfg['with_ssim'] == True:
            ssim_map = compute_ssim_loss(rgb1, rgb2)
            diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        loss = self.mean_on_mask(diff_img, valid_points)
        return loss
    def forward(self, rgb_pred, rgb_gt,  depth_pred=None, depth_gt=None, 
                t_list=None, X=None, Y=None,  rgb_pc1=None, 
                rgb_pc1_proj=None, valid_points=None, 
                d1_proj=None, d2=None, d2_proj=None, d1=None, weights={}, rgb_loss_type='l2', **kwargs):
        rgb_gt = rgb_gt.cuda()
        
        if weights['rgb_weight'] != 0.0:
            rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt, rgb_loss_type)
        else:
            rgb_full_loss = torch.tensor(0.0).cuda().float()
        if weights['depth_weight'] != 0.0:
            depth_loss = self.get_depth_loss(depth_pred, depth_gt)
        else: 
            depth_loss = torch.tensor(0.0).cuda().float()
        
        if weights['weight_dist_2nd_loss'] !=0.0 or weights['weight_dist_1st_loss'] !=0.0:
            loss_dist_1st, loss_dist_2nd = self.get_weight_dist_loss(t_list)
        else:
            loss_dist_1st, loss_dist_2nd = torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()
        if weights['pc_weight']!=0.0: 
            pc_loss = self.get_pc_loss(X, Y)
        else:
            pc_loss = torch.tensor(0.0).cuda().float()
        if weights['rgb_s_weight']!=0.0:
            rgb_s_loss = self.get_rgb_s_loss(rgb_pc1, rgb_pc1_proj, valid_points)
        else:
            rgb_s_loss = torch.tensor(0.0).cuda().float()
        if weights['depth_consistency_weight'] != 0.0:
            depth_consistency_loss = self.get_depth_consistency_loss(d1_proj, d2, d2_proj, d1)
        else: 
            depth_consistency_loss = torch.tensor(0.0).cuda().float()
        

        if (weights['rgb_weight']!=0.0) or (weights['depth_weight'] !=0.0):
            rgb_l2_mean = F.mse_loss(rgb_pred, rgb_gt)
        else:
            rgb_l2_mean =  torch.tensor(0.0).cuda().float()

        loss = weights['rgb_weight'] * rgb_full_loss + \
                   weights['depth_weight'] * depth_loss +\
                        weights['weight_dist_1st_loss'] * loss_dist_1st+\
                            weights['weight_dist_2nd_loss'] * loss_dist_2nd+\
                                weights['pc_weight'] * pc_loss+\
                                    weights['rgb_s_weight'] * rgb_s_loss+\
                                        weights['depth_consistency_weight'] * depth_consistency_loss
                                                               
        if torch.isnan(loss):
            breakpoint()
        return_dict = {
            'loss': loss,
            'loss_rgb': rgb_full_loss,
            'loss_depth': depth_loss,
            'l2_mean': rgb_l2_mean,
            'loss_dist_1st':loss_dist_1st,
            'loss_dist_2nd': loss_dist_2nd,
            'loss_pc': pc_loss,
            'loss_rgb_s': rgb_s_loss,
            'loss_depth_consistency': depth_consistency_loss
        }
     
        return return_dict



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
compute_ssim_loss = SSIM().to('cuda')