import os
from re import L
import sys
import argparse
import time
import logging
import torch

sys.path.append(os.path.join(sys.path[0], '..'))
from dataloading import get_dataloader, load_config
from model.checkpoints import CheckpointIO
from model.common import compute_errors
from model.eval_images import Eval_Images
import model as mdl
import imageio
import numpy as np
import lpips as lpips_lib
from utils_poses.align_traj import align_scale_c2b_use_a2b, align_ate_c2b_use_a2b
from tqdm import tqdm
from model.common import mse2psnr
from torch.utils.tensorboard import SummaryWriter

def eval(cfg):
    torch.manual_seed(0)
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['eval_pose']['extraction_dir'])
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)
    log_out_dir = os.path.join(out_dir, 'logs')
    writer = SummaryWriter(log_out_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(os.path.join(generation_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # logger.info(args)

    # Model
    network_type = cfg['model']['network_type']
    if network_type=='official':
        model = mdl.OfficialStaticNerf(cfg)

    rendering_cfg = cfg['rendering']
    renderer = mdl.Renderer(model, rendering_cfg, device=device)

    # init model
    nope_nerf = mdl.get_model(renderer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=nope_nerf) # changed
    # Dataloading
    train_loader, train_dataset = get_dataloader(cfg, mode='train', shuffle=False)
    eval_loader, eval_dataset = get_dataloader(cfg, mode='eval', shuffle=False)
    checkpoint_io.load(cfg['extract_images']['model_file'])
    use_learnt_poses = cfg['pose']['learn_pose']
    use_learnt_focal = cfg['pose']['learn_focal']
    num_epoch = cfg['eval_pose']['opt_pose_epoch']
    init_method = cfg['eval_pose']['init_method']
    opt_eval_lr = cfg['eval_pose']['opt_eval_lr']

    if cfg['eval_pose']['type_to_eval'] == 'train':
        N_imgs = train_dataset['img'].N_imgs
        img_list = train_dataset['img'].img_list
        loader = train_loader
        render_dir = os.path.join(generation_dir, 'eval_trained')
    else:
        N_imgs = eval_dataset['img'].N_imgs
        img_list = eval_dataset['img'].img_list
        loader = eval_loader
        render_dir = os.path.join(generation_dir, 'eval', init_method)
    
    

    if use_learnt_focal:
        focal_net = mdl.LearnFocal(cfg['pose']['learn_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'])
        checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net)
        checkpoint_io_focal.load(cfg['extract_images']['model_file_focal'])
        fxfy = focal_net(0)
        print('learned fx: {0:.2f}, fy: {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
    else:
        focal_net = None
        fxfy = None
        
    if use_learnt_poses:
        if cfg['pose']['init_pose']:
            init_pose = train_dataset['img'].c2ws # init with colmap
        else:
            init_pose = None
        learned_pose_param_net = mdl.LearnPose(train_dataset['img'].N_imgs, cfg['pose']['learn_R'], cfg['pose']['learn_t'], cfg=cfg,init_c2w=init_pose).to(device=device)
        checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=learned_pose_param_net)
        checkpoint_io_pose.load(cfg['extract_images']['model_file_pose'])
        if cfg['eval_pose']['type_to_eval'] == 'train':
            eval_pose_param_net = learned_pose_param_net
        else:
            with torch.no_grad():
                init_c2ws = eval_dataset['img'].c2ws.to(device)
                learned_c2ws_train = torch.stack([learned_pose_param_net(i) for i in range(train_dataset['img'].N_imgs_train)])
                colmap_c2ws_train = train_dataset['img'].c2ws # (N, 4, 4)
                colmap_c2ws_train = colmap_c2ws_train.to(device)
                if init_method=='scale':
                    init_c2ws, scale_colmap2est = align_scale_c2b_use_a2b(colmap_c2ws_train, learned_c2ws_train, init_c2ws.clone())
                elif init_method=='ate':
                    init_c2ws = align_ate_c2b_use_a2b(colmap_c2ws_train, learned_c2ws_train, init_c2ws)
                elif init_method=='pre':
                    sample_rate = train_dataset['img'].sample_rate
                    init_c2ws = learned_c2ws_train[int(sample_rate/2)-1::sample_rate-1][:N_imgs]
                elif init_method=='none':
                    init_c2ws = None
            eval_pose_param_net = mdl.LearnPose(eval_dataset['img'].N_imgs, learn_R=True, learn_t=True, cfg=cfg, init_c2w=init_c2ws).to(device=device)
        optimizer_eval_pose = torch.optim.Adam(eval_pose_param_net.parameters(), lr=opt_eval_lr)
        scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_eval_pose,
                                                                   milestones=list(range(0, int(num_epoch), int(num_epoch/5))),
                                                                gamma=0.5)      
        '''Optimise eval poses'''
        if cfg['eval_pose']['type_to_eval'] != 'train':
            eval_pose_cfg = cfg['eval_pose']
            trainer = mdl.Trainer_pose(nope_nerf, eval_pose_cfg, device=device, optimizer_pose=optimizer_eval_pose, 
                                    pose_param_net=eval_pose_param_net, focal_net=focal_net)
            for epoch_i in tqdm(range(num_epoch), desc='optimising eval'):
                L2_loss_epoch = []
                psnr_epoch = []
                for batch in eval_loader:
                    losses = trainer.train_step(batch)
                    L2_loss_epoch.append(losses['loss'].item())
                L2_loss_mean = np.mean(L2_loss_epoch)
                opt_pose_psnr = mse2psnr(L2_loss_mean)
                scheduler_eval_pose.step()

                writer.add_scalar('opt/psnr', opt_pose_psnr, epoch_i)

                tqdm.write('{0:6d} ep: Opt: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, L2_loss_mean, opt_pose_psnr))
    eval_pose_param_net.eval()
    eval_c2ws= torch.stack([eval_pose_param_net(i) for i in range(N_imgs)])


    # Generator
    generator = Eval_Images(
        renderer, cfg,use_learnt_poses=use_learnt_poses,
        use_learnt_focal=use_learnt_focal,
        device=device, render_type=cfg['rendering']['type'], c2ws=eval_c2ws, img_list=img_list
    )
    
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    imgs = []
    depths = []
    eval_mse_list = []
    eval_psnr_list = []
    eval_ssim_list = []
    eval_lpips_list = []
    depth_gts = []
    depth_preds = []
    # init lpips loss.
    lpips_metric = lpips_lib.LPIPS(net='vgg').to(device)
    min_depth=0.1
    max_depth=20
    for data in loader:
        out = generator.eval_images(data, render_dir, fxfy, lpips_metric, logger=logger, min_depth=min_depth, max_depth=max_depth)
        imgs.append(out['img'])
        depths.append(out['depth'])
        eval_mse_list.append(out['mse'])
        eval_psnr_list.append(out['psnr'])
        eval_ssim_list.append(out['ssim'])
        eval_lpips_list.append(out['lpips'])
        depth_preds.append(out['depth_pred'])
        depth_gts.append(out['depth_gt'])

    mean_mse = np.mean(eval_mse_list)
    mean_psnr = np.mean(eval_psnr_list)
    mean_ssim = np.mean(eval_ssim_list)
    mean_lpips = np.mean(eval_lpips_list)
    print('--------------------------')
    print('Mean MSE: {0:.2f}, PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(mean_mse, mean_psnr,
                                                                                    mean_ssim, mean_lpips))

    print("{0:.2f}".format(mean_psnr),'&' "{0:.2f}".format(mean_ssim), '&', "{0:.2f}".format(mean_lpips))     
   
    if cfg['extract_images']['eval_depth']:
        depth_errors = []
        ratio = np.median(np.concatenate(depth_gts)) / \
                        np.median(np.concatenate(depth_preds))
        for i in range(len(depth_preds)):
            gt_depth = depth_gts[i]
            pred_depth = depth_preds[i]

            pred_depth *= ratio
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth

            depth_errors.append(compute_errors(gt_depth, pred_depth))
        

        mean_errors = np.array(depth_errors).mean(0)                                                                                       
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

        with open(os.path.join(generation_dir, 'depth_evaluation.txt'), 'a') as f:
            f.writelines(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3") + '\n')
            f.writelines(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    imgs = np.stack(imgs, axis=0)
    video_out_dir = os.path.join(render_dir, 'video_out')
    if not os.path.exists(video_out_dir):
        os.makedirs(video_out_dir)
    imageio.mimwrite(os.path.join(video_out_dir, 'img.mp4'), imgs, fps=30, quality=9)

if __name__=='__main__':
    # Config
    parser = argparse.ArgumentParser(
        description='Extract images.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    eval(cfg)


       
