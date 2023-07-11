import os
import sys
import logging
import time
import argparse

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import dataloading as dl
import model as mdl
from utils_poses.comp_ate import compute_ATE, compute_rpe
from model.common import backup,  mse2psnr
from utils_poses.align_traj import align_ate_c2b_use_a2b
def train(cfg):
    logger_py = logging.getLogger(__name__)

    # # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    # params
    out_dir = cfg['training']['out_dir']
    backup_every = cfg['training']['backup_every']
    
    lr = cfg['training']['learning_rate']

    mode = cfg['training']['mode']
    train_loader, train_dataset = dl.get_dataloader(cfg, mode=mode, shuffle=cfg['dataloading']['shuffle'])
    test_loader, _ = dl.get_dataloader(cfg, mode=mode, shuffle=cfg['dataloading']['shuffle'])
    iter_test = iter(test_loader)
    data_test = next(iter_test)
    

    n_views = train_dataset['img'].N_imgs
    # init network
    network_type = cfg['model']['network_type']
    auto_scheduler = cfg['training']['auto_scheduler']
    scheduling_epoch = cfg['training']['scheduling_epoch']
    

    if network_type=='official':
        model = mdl.OfficialStaticNerf(cfg)
    
     # init renderer 
    rendering_cfg = cfg['rendering']
    renderer = mdl.Renderer(model, rendering_cfg, device=device)
    # init model
    nope_nerf = mdl.get_model(renderer, cfg, device=device)
    # init optimizer
    weight_decay = cfg['training']['weight_decay']
    optimizer = optim.Adam(nope_nerf.parameters(), lr=lr, weight_decay=weight_decay)

    # init checkpoints and load
    checkpoint_io = mdl.CheckpointIO(out_dir, model=nope_nerf, optimizer=optimizer)
    load_dir = cfg['training']['load_dir']

    try:
        load_dict = checkpoint_io.load(load_dir, load_model_only=cfg['training']['load_ckpt_model_only'])
    except FileExistsError:
        load_dict = dict()
        
    # resume training
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
    'loss_val_best', -np.inf)
    patient_count = load_dict.get('patient_count', 0)
    scheduling_start = load_dict.get('scheduling_start', cfg['training']['scheduling_start'])

    if not auto_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=list(range(scheduling_start, scheduling_epoch+scheduling_start, 10)),
        gamma=cfg['training']['scheduler_gamma'], last_epoch=epoch_it)
    
    

    # init camera extrinsics
    if cfg['pose']['learn_pose']:
        if cfg['pose']['init_pose']:
            if cfg['pose']['init_pose_type']=='gt':
                init_pose = train_dataset['img'].c2ws 
            elif cfg['pose']['init_pose_type']=='colmap':
                init_pose = train_dataset['img'].c2ws_colmap
            init_pose = init_pose.to(device)
        else:
            init_pose = None
            
        pose_param_net = mdl.LearnPose(n_views, cfg['pose']['learn_R'], 
                            cfg['pose']['learn_t'], cfg, init_c2w=init_pose).to(device=device)
        
        optimizer_pose = optim.Adam(pose_param_net.parameters(), lr=cfg['training']['pose_lr'])
        checkpoint_io_pose = mdl.CheckpointIO(out_dir, model=pose_param_net, optimizer=optimizer_pose)
        try:
            pose_load_dir = cfg['training']['load_pose_dir']
            load_dict = checkpoint_io_pose.load(pose_load_dir, load_model_only=cfg['training']['load_ckpt_model_only'])
        except FileExistsError:
            load_dict = dict()
        epoch_it = load_dict.get('epoch_it', -1)
        if not auto_scheduler:
            scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, 
                                                                milestones=list(range(scheduling_start, scheduling_epoch+scheduling_start, 100)),
                                                                gamma=cfg['training']['scheduler_gamma_pose'], last_epoch=epoch_it)
    else:
        optimizer_pose = None
        pose_param_net = None
    # init distortion parameters
    if cfg['distortion']['learn_distortion']:
        distortion_net = mdl.Learn_Distortion(n_views, cfg['distortion']['learn_scale'], cfg['distortion']['learn_shift'], cfg).to(device=device)
        optimizer_distortion = optim.Adam(distortion_net.parameters(), lr=cfg['training']['distortion_lr'])
        checkpoint_io_distortion = mdl.CheckpointIO(out_dir, model=distortion_net, optimizer=optimizer_distortion)
        try:
            distortion_load_dir = cfg['training']['load_distortion_dir']
            load_dict = checkpoint_io_distortion.load(distortion_load_dir, load_model_only=cfg['training']['load_ckpt_model_only'])
        except FileExistsError:
            load_dict = dict()
        epoch_it = load_dict.get('epoch_it', -1)
        if not auto_scheduler:
            scheduler_distortion = torch.optim.lr_scheduler.MultiStepLR(optimizer_distortion, 
                                                                    milestones=list(range(scheduling_start, 10000+scheduling_start, 100)),
                                                                    gamma=cfg['training']['scheduler_gamma_distortion'], last_epoch=epoch_it)
    else:
        optimizer_distortion = None
        distortion_net = None

    # init intrinsics 
    if cfg['pose']['learn_focal']:
        if cfg['pose']['init_focal_type']=='gt':
            init_focal=[train_dataset['img'].K[0, 0], -train_dataset['img'].K[1, 1]]
        else: 
            init_focal = None
        focal_net = mdl.LearnFocal(cfg['pose']['update_focal'], cfg['pose']['fx_only'], order=cfg['pose']['focal_order'], init_focal=init_focal).to(device=device)
        optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=cfg['training']['focal_lr'])
        checkpoint_io_focal = mdl.CheckpointIO(out_dir, model=focal_net, optimizer=optimizer_focal)
        try:
            focal_load_dir = cfg['training']['load_focal_dir']
            load_dict = checkpoint_io_focal.load(focal_load_dir)
        except FileExistsError:
            load_dict = dict()
        epoch_it = load_dict.get('epoch_it', -1)
        if not auto_scheduler:
            scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=list(range(scheduling_start, scheduling_epoch+scheduling_start, 100)),
                                                            gamma=cfg['training']['scheduler_gamma_focal'], last_epoch=epoch_it)
    else:
        optimizer_focal = None
        focal_net = None
   
     # init training
    training_cfg = cfg['training']
    trainer = mdl.Trainer(nope_nerf, optimizer, training_cfg, device=device, optimizer_pose=optimizer_pose, 
                        pose_param_net=pose_param_net, optimizer_focal=optimizer_focal,focal_net=focal_net,
                        optimizer_distortion=optimizer_distortion,distortion_net=distortion_net, cfg_all=cfg
                        )

    
    

    logger = SummaryWriter(os.path.join(out_dir, 'logs'))
        
    # init training output
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    visualize_every = cfg['training']['visualize_every']
    validate_every = cfg['training']['validate_every']
    eval_pose_every = cfg['training']['eval_pose_every']
    eval_img_every = cfg['training']['eval_img_every']

    render_path = os.path.join(out_dir, 'rendering')
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    


    # Print model
    nparameters = sum(p.numel() for p in nope_nerf.parameters())
    logger_py.info(nope_nerf)
    logger_py.info('Total number of parameters: %d' % nparameters)
    t0b = time.time()

    
    patient = cfg['training']['patient']
    length_smooth=cfg['training']['length_smooth']
    scheduling_mode = cfg['training']['scheduling_mode']
    psnr_window = []

    # torch.autograd.set_detect_anomaly(True)

    log_scale_shift_per_view = cfg['training']['log_scale_shift_per_view']
    scale_dict = {}
    shift_dict = {}
    # load gt poses for evaluation
    if eval_pose_every>0:
        gt_poses = train_dataset['img'].c2ws.to(device) 
    # for epoch_it in tqdm(range(epoch_start+1, exit_after), desc='epochs'):
    while epoch_it < (scheduling_start + scheduling_epoch):
        epoch_it +=1
        L2_loss_epoch = []
        pc_loss_epoch = []
        rgb_s_loss_epoch = []
        for batch in train_loader:
            it += 1
            idx = batch.get('img.idx')
            loss_dict = trainer.train_step(batch, it, epoch_it, scheduling_start, render_path)
            loss = loss_dict['loss']
            L2_loss_epoch.append(loss_dict['l2_mean'].item())
            pc_loss_epoch.append(loss_dict['loss_pc'].item())
            rgb_s_loss_epoch.append(loss_dict['loss_rgb_s'].item())
            scale_dict['view %02d' % (idx)] = loss_dict['scale']
            shift_dict['view %02d' % (idx)] = loss_dict['shift']
            if print_every > 0 and (it % print_every) == 0:
                tqdm.write('[Epoch %02d] it=%03d, loss=%.8f, time=%.4f'
                            % (epoch_it, it, loss, time.time() - t0b))
                logger_py.info('[Epoch %02d] it=%03d, loss=%.4f, time=%.4f'
                            % (epoch_it, it, loss, time.time() - t0b))
                t0b = time.time()
                for l, num in loss_dict.items():
                    logger.add_scalar('train/'+l, num.detach().cpu(), it)
                if log_scale_shift_per_view:
                    for l, num in scale_dict.items():
                        logger.add_scalar('train/scale'+l, num, it)
                    for l, num in shift_dict.items():
                        logger.add_scalar('train/shift'+l, num, it)
            
            if visualize_every > 0 and (it % visualize_every)==0:
                logger_py.info("Rendering")
                out_render_path = os.path.join(render_path, '%04d_vis' % it)
                if not os.path.exists(out_render_path):
                    os.makedirs(out_render_path)
                val_rgb = trainer.render_visdata(
                            data_test, 
                            cfg['training']['vis_resolution'], 
                            it, out_render_path)
                #logger.add_image('rgb', val_rgb, it)
            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(test_loader)

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)
        
            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                logger_py.info('Saving checkpoint')
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best, scheduling_start=scheduling_start, patient_count=patient_count)
                if cfg['pose']['learn_pose']:
                    checkpoint_io_pose.save('model_pose.pt', epoch_it=epoch_it, it=it)
                if cfg['pose']['learn_focal']:
                    checkpoint_io_focal.save('model_focal.pt', epoch_it=epoch_it, it=it)
                if cfg['distortion']['learn_distortion']:
                    checkpoint_io_distortion.save('model_distortion.pt', epoch_it=epoch_it, it=it)

            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                logger_py.info('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val_best, scheduling_start=scheduling_start, patient_count=patient_count)
                if cfg['pose']['learn_pose']:
                    checkpoint_io_pose.save('model_pose_%d.pt' % it, epoch_it=epoch_it, it=it)
                if cfg['pose']['learn_focal']:
                    checkpoint_io_focal.save('model_focal_%d.pt' % it, epoch_it=epoch_it, it=it)
                if cfg['distortion']['learn_distortion']:
                    checkpoint_io_distortion.save('model_distortion_%d.pt' % it, epoch_it=epoch_it, it=it)

        pc_loss_epoch = np.mean(pc_loss_epoch)
        logger.add_scalar('train/loss_pc_epoch', pc_loss_epoch, it) 
        rgb_s_loss_epoch = np.mean(rgb_s_loss_epoch) 
        logger.add_scalar('train/loss_rgbs_epoch', rgb_s_loss_epoch, it)  
        if (eval_pose_every>0 and (epoch_it % eval_pose_every) == 0):
            with torch.no_grad():
                learned_poses = torch.stack([pose_param_net(i) for i in range(n_views)])
            c2ws_est_aligned = align_ate_c2b_use_a2b(learned_poses, gt_poses)
            ate = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
            rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
            tqdm.write('{0:6d} ep: Train: ATE: {1:.3f} RPE_r: {2:.3f}'.format(epoch_it, ate, rpe_rot* 180 / np.pi))
            eval_dict = {
                'ate_trans': ate,
                'rpe_trans': rpe_trans*100,
                'rpe_rot': rpe_rot* 180 / np.pi
            }
            for l, num in eval_dict.items():
                logger.add_scalar('eval/'+l, num, it)
        if (eval_img_every>0 and (epoch_it % eval_img_every) == 0):    
            L2_loss_mean = np.mean(L2_loss_epoch)
            psnr = mse2psnr(L2_loss_mean)
            tqdm.write('{0:6d} ep: Train: PSNR: {1:.3f}'.format(epoch_it, psnr))
            logger.add_scalar('train/psnr', psnr, it)
            
        if not auto_scheduler:
            scheduler.step()
            new_lr = scheduler.get_lr()[0]
            if cfg['pose']['learn_pose']:
                scheduler_pose.step()
                new_lr_pose = scheduler_pose.get_lr()[0]
            if cfg['pose']['learn_focal']:
                scheduler_focal.step()
                new_lr_focal = scheduler_focal.get_lr()[0]
            if cfg['distortion']['learn_distortion']:
                scheduler_distortion.step()
                new_lr_distortion = scheduler_distortion.get_lr()[0]
        else:
            psnr_window.append(psnr)
            if len(psnr_window) >= length_smooth:
                psnr_window = psnr_window[-length_smooth:]
                metric_val = np.array(psnr_window).mean()
                if (metric_val - metric_val_best) >= 0:
                    metric_val_best = metric_val
                else:
                    patient_count = patient_count + 1
                    if (patient_count == patient):
                        scheduling_start = epoch_it
            if epoch_it < scheduling_start:
                new_lr = cfg['training']['learning_rate']
                new_lr_pose = cfg['training']['pose_lr']
                new_lr_focal = cfg['training']['focal_lr']
                new_lr_distortion = cfg['training']['distortion_lr']
            else:
                new_lr = cfg['training']['learning_rate'] * ((cfg['training']['scheduler_gamma'])**int((epoch_it-scheduling_start)/10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                if cfg['pose']['learn_pose']:
                    new_lr_pose = cfg['training']['pose_lr'] * ((cfg['training']['scheduler_gamma_pose'])**int((epoch_it-scheduling_start)/100))
                    for param_group in optimizer_pose.param_groups:
                        param_group['lr'] = new_lr_pose
                if cfg['pose']['learn_focal']:
                    new_lr_focal = cfg['training']['focal_lr'] * ((cfg['training']['scheduler_gamma_focal'])**int((epoch_it-scheduling_start)/100))
                    for param_group in optimizer_focal.param_groups:
                        param_group['lr'] = new_lr_focal
                if cfg['distortion']['learn_distortion']:
                    new_lr_distortion = cfg['training']['distortion_lr'] * ((cfg['training']['scheduler_gamma_distortion'])**int((epoch_it-scheduling_start)/100))
                    for param_group in optimizer_distortion.param_groups:
                        param_group['lr'] = new_lr_distortion
        if scheduling_mode=='reset' and epoch_it == scheduling_start:
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

        logger.add_scalar('train/lr', new_lr, it)
        if cfg['pose']['learn_pose']:
            logger.add_scalar('train/lr_pose', new_lr_pose, it)
        if cfg['pose']['learn_focal']:
            logger.add_scalar('train/lr_focal', new_lr_focal, it)
        if cfg['distortion']['learn_distortion']:
            logger.add_scalar('train/lr_distortion', new_lr_distortion, it)

if __name__=='__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Training of nope-nerf model'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    cfg = dl.load_config(args.config, 'configs/default.yaml')
    # backup model
    backup(cfg['training']['out_dir'], args.config)
    train(cfg=cfg)
    