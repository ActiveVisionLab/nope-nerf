import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import (
    get_mask, image_points_to_world, origin_to_world, transform_to_world, convert2mip)
from .common import get_ndc_rays_fxfy

epsilon = 1e-6
class Renderer(nn.Module):
    def __init__(self, model, cfg, device=None,
                 **kwargs):
        super().__init__()
        self._device = device
        self.depth_range = cfg['depth_range']
        self.n_max_network_queries = cfg['n_max_network_queries']
        self.white_background = cfg['white_background']
        self.cfg= cfg
        self.model = model.to(device)


    def forward(self, pixels, depth, camera_mat, world_mat, scale_mat, 
                      rendering_technique, add_noise=True, eval_=False,
                     it=1000000):
        if rendering_technique == 'nope_nerf':
            out_dict = self.nope_nerf(
                pixels, depth, camera_mat, world_mat, 
                scale_mat, it=it, add_noise=add_noise, eval_=eval_
            )
        elif rendering_technique == 'phong_renderer':
            out_dict = self.phong_renderer(
                pixels, camera_mat, world_mat, scale_mat, it=it
            )
        return out_dict
   
    def nope_nerf(self, pixels, depth, camera_mat, world_mat, 
                scale_mat, add_noise=False, it=100000, eval_=False):
        # Get configs
        batch_size, n_points, _ = pixels.shape
        device = self._device
        full_steps = self.cfg['num_points']
        dist_alpha = self.cfg['dist_alpha']
        sample_option = self.cfg['sample_option']
        use_dir = self.cfg['use_ray_dir']
        normalise_ray = self.cfg['normalise_ray']
        normal_loss = self.cfg['normal_loss']
        outside_steps = self.cfg['outside_steps']


        depth_range = torch.tensor(self.depth_range)
        n_max_network_queries = self.n_max_network_queries

        # Find surface points in world coorinate
        camera_world = origin_to_world(
                n_points, camera_mat, world_mat, scale_mat
            )
        points_world = transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat)
        

        d_i_gt = torch.norm(points_world - camera_world, p=2, dim=-1)
        
        # Prepare camera projection
        pixels_world = image_points_to_world(
            pixels, camera_mat, world_mat,scale_mat
        )
        ray_vector = (pixels_world - camera_world)
        ray_vector_norm = ray_vector.norm(2,2)
        if normalise_ray:
            ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1) # normalised ray vector
        else:
            d_i_gt = d_i_gt / ray_vector_norm # used for guide sampling, convert dist to depth
        
        d_i = d_i_gt.clone()
        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0

        # Get mask for predicted depth
        mask_pred = get_mask(d_i)
        
        # with torch.no_grad():
        dists = torch.ones_like(d_i).to(device)
        dists[mask_pred] = d_i[mask_pred]
        dists[mask_zero_occupied] = 0.
        network_object_mask = mask_pred & ~mask_zero_occupied
      
        network_object_mask = network_object_mask[0]
        dists = dists[0]

        # Project depth to 3d poinsts
        camera_world = camera_world.reshape(-1, 3)
        ray_vector = ray_vector.reshape(-1, 3)
       
        points = camera_world + ray_vector * dists.unsqueeze(-1)
        points = points.view(-1,3)
        z_val = torch.linspace(0., 1., steps=full_steps-outside_steps, device=device)
        z_val = z_val.view(1, 1, -1).repeat(batch_size, n_points, 1)
        
        if sample_option=='ndc':
            z_val, pts, ray_vector_fg = self.sample_ndc(camera_mat, camera_world, ray_vector, z_val, depth_range=[0., 1.])
        elif sample_option=='uniform':
            z_val, pts, ray_vector_fg = self.sample_uniform(camera_world, ray_vector, z_val, add_noise, depth_range)
  
        if not use_dir:
            ray_vector_fg = torch.ones_like(ray_vector_fg)
        # Run Network
        noise = not eval_
        rgb_fg, logits_alpha_fg = [], []
        for i in range(0, pts.shape[0], n_max_network_queries):
            rgb_i, logits_alpha_i= self.model(
                pts[i:i+n_max_network_queries], 
                ray_vector_fg[i:i+n_max_network_queries], 
                return_addocc=True, noise=noise, it=it
            )
            rgb_fg.append(rgb_i)
            logits_alpha_fg.append(logits_alpha_i)
        rgb_fg = torch.cat(rgb_fg, dim=0)
        logits_alpha_fg = torch.cat(logits_alpha_fg, dim=0)
        
        rgb = rgb_fg.reshape(batch_size * n_points, full_steps, 3)
        alpha = logits_alpha_fg.view(batch_size * n_points, full_steps)
        
        if dist_alpha:
            t_vals = z_val.view(batch_size * n_points, full_steps)
            deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (H, W, N_sample-1)
            dist_far = torch.empty(size=(batch_size * n_points, 1), dtype=torch.float32, device=dists.device).fill_(1e10)  # (H, W, 1)
            deltas = torch.cat([deltas, dist_far], dim=-1)  # (H, W, N_sample)
            alpha = 1 - torch.exp(-1.0 * alpha * deltas)  # (H, W, N_sample)      
            alpha[:, -1] = 1. # enforce predicted depth>0

        weights = alpha * torch.cumprod(torch.cat([torch.ones((rgb.shape[0], 1), device=device), 1.-alpha + epsilon ], -1), -1)[:, :-1]
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        dist_pred = torch.sum(weights.unsqueeze(-1) * z_val, dim=-2).squeeze(-1)
        if not eval_ and normal_loss:
            surface_mask = network_object_mask.view(-1)
            surface_points = points[surface_mask]
            N = surface_points.shape[0]
            surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01      
            pp = torch.cat([surface_points, surface_points_neig], dim=0)
            g = self.model.gradient(pp, it) 
            normals_ = g[:, 0, :] / (g[:, 0, :].norm(2, dim=1).unsqueeze(-1) + 10**(-5))
            diff_norm = torch.norm(normals_[:N] - normals_[N:], dim=-1)
        else:
            diff_norm = None

        if self.white_background:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map.unsqueeze(-1))

        d_i_gt =  d_i_gt[0] 
        if eval_ and normalise_ray:
            # print('-------normalising depth-------')
            dist_pred = dist_pred / ray_vector_norm[0] # change dist to depth, consistent with gt depth for evaluation
            dists = dists / ray_vector_norm[0]
            d_i_gt = d_i_gt / ray_vector_norm[0]
        dist_rendered_masked = dist_pred[network_object_mask]
        dist_dpt_masked = d_i_gt[network_object_mask]
        if sample_option=='ndc':
            dist_dpt_masked = 1 - 1/dist_dpt_masked
        out_dict = {
            'rgb': rgb_values.reshape(batch_size, -1, 3),
            'z_vals': z_val.squeeze(-1),
            'normal': diff_norm,
            'depth_pred': dist_rendered_masked, # for loss
            'depth_gt': dist_dpt_masked,  # for loss
            'alpha': alpha
        }
        return out_dict
    def sample_ndc(self, camera_mat, camera_world, ray_vector, z_val, depth_range=[0., 1.]):
        batch_size, n_points, full_steps = z_val.shape
        focal = torch.cat([camera_mat[:, 0, 0], camera_mat[:, 1, 1]])
        ray_ori_world, ray_dir_world = get_ndc_rays_fxfy(focal, 1.0, rays_o=camera_world,
                                                    rays_d=ray_vector)
        z_val = depth_range[0] * (1. - z_val) + depth_range[1]* z_val
        pts = ray_ori_world.unsqueeze(-2) \
            + ray_dir_world.unsqueeze(-2) * z_val.unsqueeze(-1)
        pts = pts.reshape(-1, 3)
        ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
        ray_vector_fg = -1*ray_vector_fg.reshape(-1, 3)
        z_val = z_val.view(-1, full_steps, 1)
        return z_val, pts, ray_vector_fg
   
    def sample_uniform(self, camera_world, ray_vector, z_val, add_noise, depth_range):
        batch_size, n_points, full_steps = z_val.shape
        z_val = depth_range[0] * (1. - z_val) + depth_range[1]* z_val
        if add_noise:
            di_mid = .5 * (z_val[:, :, 1:] + z_val[:, :, :-1])
            di_high = torch.cat([di_mid, z_val[:, :, -1:]], dim=-1)
            di_low = torch.cat([z_val[:, :, :1], di_mid], dim=-1)
            noise = torch.rand(batch_size, n_points, full_steps, device=self._device)
            z_val = di_low + (di_high - di_low) * noise 
        pts = camera_world.unsqueeze(-2) \
            + ray_vector.unsqueeze(-2) * z_val.unsqueeze(-1)
        pts = pts.reshape(-1, 3)
        ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
        ray_vector_fg = -1*ray_vector_fg.reshape(-1, 3)
        z_val = z_val.view(-1, full_steps, 1)
        return z_val, pts, ray_vector_fg
    



    def phong_renderer(self, pixels, camera_mat, world_mat, 
                     scale_mat, it):
        batch_size, num_pixels, _ = pixels.shape
        device = self._device
        rad = self.cfg['radius']
        n_points = num_pixels
        # fac = self.cfg['sig_factor']
        pixels_world = image_points_to_world(pixels, camera_mat, world_mat,scale_mat)
        camera_world = origin_to_world(num_pixels, camera_mat, world_mat, scale_mat)
        ray_vector = (pixels_world - camera_world)
        ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1)

        light_source = camera_world[0,0] 
        #torch.Tensor([ 1.4719,  0.0284, -1.9837]).cuda().float()# # torch.Tensor([ 0.3074, -0.8482, -0.1880]).cuda().float() #camera_world[0,0] #torch.Tensor([ 1.4719,  0.0284, -1.9837]).cuda().float()
        light = (light_source / light_source.norm(2)).unsqueeze(1).cuda()
    
        diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
        ambiant = torch.Tensor([0.3,0.3,0.3]).float()


        # run ray tracer / depth function --> 3D point on surface (differentiable)
        self.model.eval()
        with torch.no_grad():
            d_i = self.ray_marching(camera_world, ray_vector, self.model,
                                         n_secant_steps=8,  n_steps=[int(512),int(512)+1], rad=rad)
        # Get mask for where first evaluation point is occupied
        d_i = d_i.detach()
    
        mask_zero_occupied = d_i == 0
        mask_pred = get_mask(d_i).detach()

        # For sanity for the gradients
        with torch.no_grad():
            dists =  torch.ones_like(d_i).to(device)
            dists[mask_pred] = d_i[mask_pred].detach()
            dists[mask_zero_occupied] = 0.
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask[0]
            dists = dists[0]

            camera_world = camera_world.reshape(-1, 3)
            ray_vector = ray_vector.reshape(-1, 3)

            points = camera_world + ray_vector * dists.unsqueeze(-1)
            points = points.view(-1,3)
            view_vol = -1 * ray_vector.view(-1, 3)
            rgb_values = torch.ones_like(points).float().cuda()

            surface_points = points[network_object_mask]
            surface_view_vol = view_vol[network_object_mask]

            # Derive Normals
            grad = []
            for pnts in torch.split(surface_points, 1000000, dim=0):
                grad.append(self.model.gradient(pnts, it)[:,0,:].detach())
                torch.cuda.empty_cache()
            grad = torch.cat(grad,0)
            surface_normals = grad / grad.norm(2,1,keepdim=True)

        diffuse = torch.mm(surface_normals, light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0).cuda()
        rgb_values[network_object_mask] = (ambiant.unsqueeze(0).cuda() + diffuse).clamp_max(1.0)

        with torch.no_grad():
            rgb_val = torch.zeros(batch_size * n_points, 3, device=device)
            rgb_val[network_object_mask] = self.model(surface_points, surface_view_vol)

        out_dict = {
            'rgb': rgb_values.reshape(batch_size, -1, 3),
            'normal': None,
            'rgb_surf': rgb_val.reshape(batch_size, -1, 3),
        }

        return out_dict


    def ray_marching(self, ray0, ray_direction, model, c=None,
                             tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                             depth_range=[0., 2.4], max_points=3500000, rad=1.0):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        '''
        # Shotscuts
        batch_size, n_pts, D = ray0.shape
        device = ray0.device
        tau = 0.5
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()

            
        depth_intersect, _ = get_sphere_intersection(ray0[:,0], ray_direction, r=rad)
        d_intersect = depth_intersect[...,1]            
        
        d_proposal = torch.linspace(
            0, 1, steps=n_steps).view(
                1, 1, n_steps, 1).to(device)
        d_proposal = depth_range[0] * (1. - d_proposal) + d_intersect.view(1, -1, 1,1)* d_proposal

        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
            ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

        # Evaluate all proposal points in parallel
        with torch.no_grad():
            val = torch.cat([(
                self.model(p_split, only_occupancy=True) - tau)
                for p_split in torch.split(
                    p_proposal.reshape(batch_size, -1, 3),
                    int(max_points / batch_size), dim=1)], dim=1).view(
                        batch_size, -1, n_steps)

        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] < 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # write c in pointwise format
        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        
        # Apply surface depth refinement step (e.g. Secant method)
        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        if ray0_masked.shape[0] != 0:
            d_pred = self.secant(
                f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
                ray_direction_masked, tau)
            d_pred_out[mask] = d_pred

        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        return d_pred_out

    def secant(self, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, it=0):
        ''' Runs the secant method for interval [d_low, d_high].

        Args:
            d_low (tensor): start values for the interval
            d_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                f_mid = self.model(p_mid,  batchwise=False,
                                only_occupancy=True, it=it)[...,0] - tau
            ind_low = f_mid < 0
            ind_low = ind_low
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    
    def transform_to_homogenous(self, p):
        device = self._device
        batch_size, num_points, _ = p.size()
        r = torch.sqrt(torch.sum(p**2, dim=2, keepdim=True))
        p_homo = torch.cat((p, torch.ones(batch_size, num_points, 1).to(device)), dim=2) / r
        return p_homo


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape
    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect