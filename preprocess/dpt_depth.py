import os
import sys
import argparse
import torch
import cv2
import numpy as np
import imageio

sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
import model as mdl

def dpt_depth(cfg, depth_save_dir):
    torch.manual_seed(0)
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # Model
    network_type = cfg['model']['network_type']
    if network_type=='official':
        model = mdl.OfficialStaticNerf(cfg)
    rendering_cfg = cfg['rendering']
    renderer = mdl.Renderer(model, rendering_cfg, device=device)
    nope_nerf = mdl.get_model(renderer, cfg, device=device)

    # Dataloading
    train_loader, train_dataset = get_dataloader(cfg, mode='all', shuffle=False)
    nope_nerf.eval()
    DPT_model = nope_nerf.depth_estimator.to(device)


    if not os.path.exists(depth_save_dir):
        os.makedirs(depth_save_dir)
    img_list = train_dataset['img'].img_list
    
    for data in train_loader:   
        img_normalised = data.get('img.normalised_img').to(device)
        idx = data.get('img.idx')
        img_name = img_list[idx]
        depth = DPT_model(img_normalised)
        np.savez(os.path.join(depth_save_dir, 'depth_{}.npz'.format(img_name.split('.')[0])), pred=depth.detach().cpu())
        depth_array = depth[0].detach().cpu().numpy()
        imageio.imwrite(os.path.join(
            depth_save_dir, 
            '{}.png'.format(img_name.split('.')[0])), 
            np.clip(255.0 / depth_array.max() * (depth_array - depth_array.min()), 0, 255).astype(np.uint8))        
 
                                                                
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess.'
    )
    parser.add_argument('config', type=str,default='configs/preprocess.yaml', help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    if cfg['dataloading']['crop_size'] !=0:
        folder_name= 'dpt_' + str(cfg['dataloading']['crop_size'])
    else:
        folder_name = 'dpt'
    depth_save_dir = os.path.join(cfg['dataloading']['path'], cfg['dataloading']['scene'][0], folder_name)
    dpt_depth(cfg, depth_save_dir)