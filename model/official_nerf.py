import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F

class OfficialStaticNerf(nn.Module):
    def __init__(self, cfg):
        super(OfficialStaticNerf, self).__init__()
        D = cfg['model']['hidden_dim']
        pos_enc_levels = cfg['model']['pos_enc_levels']
        dir_enc_levels = cfg['model']['dir_enc_levels']
        pos_in_dims = (2 * pos_enc_levels + 1) * 3  # (2L + 0 or 1) * 3
        dir_in_dims = (2 * dir_enc_levels + 1) * 3  # (2L + 0 or 1) * 3
        self.white_bkgd = cfg['rendering']['white_background']
        self.dist_alpha = cfg['rendering']['dist_alpha']
        self.occ_activation = cfg['model']['occ_activation']

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # short cut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, 3)

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.sigmoid = nn.Sigmoid()
        if self.white_bkgd:
            self.fc_rgb.bias.data = torch.tensor([0.8, 0.8, 0.8]).float()
        else:
            self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()
        
    def gradient(self, p, it):
        with torch.enable_grad():
            p.requires_grad_(True)
            _, y = self.infer_occ(p)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return -gradients.unsqueeze(1)

    def infer_occ(self, p):
        pos_enc = encode_position(p, levels=10, inc_input=True)
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)  # (H, W, N_sample, 1)
        return x, density

    def forward(self, p, ray_d=None, only_occupancy=False, return_logits=False,return_addocc=False, 
        noise=False, it=100000, **kwargs):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims)
        :param dir_enc: (H, W, N_sample, dir_in_dims)
        :return: rgb_density (H, W, N_sample, 4)
        """
        x, density = self.infer_occ(p)
        if self.occ_activation=='softplus':
            density = F.softplus(density)
        else:
            density = density.relu()

        if not self.dist_alpha:
            density = 1 - torch.exp(-1.0 * density)
        if only_occupancy:
            return density
        elif ray_d is not None:
            dir_enc = encode_position(ray_d, levels=4, inc_input=True)
            feat = self.fc_feature(x)  # (H, W, N_sample, D)
            x = torch.cat([feat, dir_enc], dim=-1)  # (H, W, N_sample, D+dir_in_dims)
            x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
            rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)
            rgb = self.sigmoid(rgb)
            if return_addocc:
                return rgb, density
            else:
                return rgb
        

def encode_position(input, levels, inc_input):
    """
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.

    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))


