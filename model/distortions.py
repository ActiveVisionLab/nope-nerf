import torch
import torch.nn as nn

class Learn_Distortion(nn.Module):
    def __init__(self, num_cams, learn_scale, learn_shift, cfg):
        """depth distortion parameters

        Args:
            num_cams (int): number of cameras
            learn_scale (bool): whether to update scale
            learn_shift (bool): whether to update shift
            cfg (dict): argument options
        """
        super(Learn_Distortion, self).__init__()
        self.global_scales=nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=learn_scale)
        self.global_shifts=nn.Parameter(torch.zeros(size=(num_cams, 1), dtype=torch.float32), requires_grad=learn_shift)
        self.fix_scaleN = cfg['distortion']['fix_scaleN']
        self.num_cams = num_cams
    def forward(self, cam_id):
        scale = self.global_scales[cam_id]
        if scale<0.01:
            scale = torch.tensor(0.01, device=self.global_scales.device)
        if self.fix_scaleN and cam_id ==(self.num_cams-1):
            scale = torch.tensor(1, device=self.global_scales.device)
        shift = self.global_shifts[cam_id]
        
        return scale, shift