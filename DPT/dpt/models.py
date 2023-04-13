from typing import Any, List
import string
from xmlrpc.client import Boolean
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
# from pytorch_lightning import LightningModule

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
import sys
def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DepthLoss(nn.Module):
    def __init__(self, loss_type):
        """Calculate depth loss with masking scheme.
        Remove zero/negative target values
        
        Args:
            cfg (eDict): loss configuration
                - loss_type (str): the method of calculating loss
                    - smL1
                    - L1
                    - L2
                - use_inv_depth (bool): use inverse depth
        """
        super(DepthLoss, self).__init__()
        self.loss_type = loss_type
        self.use_inv_depth = False
        self.eps = 1e-6

    def forward(self, pred, target):
        """
        Args:
            pred (Nx1xHxW): predicted depth map
            target (Nx1xHxW): GT depth map
            
        Returns:
            total_loss (dict): loss items
        """
        losses = {}

        # compute mask
        non_zero_mask = target > 0
        mask = non_zero_mask

        # use inverse depth
        if self.use_inv_depth:
            target = 1. / (target + self.eps)
            pred = 1. / (pred + self.eps)
        
        if len(target[mask]) != 0:
            # compute loss
            if self.loss_type in ['smL1', 'L1', 'L2']:
                diff = target[mask] - pred[mask]
                if self.loss_type == 'smL1':
                    loss = ((diff / 2 )**2 + 1 ).pow(0.5) - 1
                elif self.loss_type == 'L1':
                    loss = diff.abs()
                elif self.loss_type == "L2":
                    loss = diff ** 2
                depth_loss = loss.mean()
            elif self.loss_type in ['eigen']:
                diff = torch.log(target[mask]) - torch.log(pred[mask])
                loss1 = (diff**2).mean()
                loss2 = (diff.sum())**2/(len(diff)**2)
                depth_loss = loss1 + 0.5 * loss2

        else:
            ### set depth_loss to 0 ###
            depth_loss = (pred*0).sum()

        return depth_loss


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        freeze=True
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


        if freeze:
            for name, p in self.named_parameters():
                p.requires_grad = False
                

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        return out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=True, freeze=True, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert
        
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        # modified head
        # head = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=4, mode="bilinear", align_corners=True),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        #     nn.Identity(),
        # )

        super().__init__(head, freeze=freeze, **kwargs)

        if path is not None:
            self.load(path)
        
        # if freeze:
        #     for name, p in self.named_parameters():
        #         print(name)
        #         p.requires_grad = False


    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


# class LitDPTModule(LightningModule):

#     def __init__(
#         self,
#         path: string = None,
#         non_negative: bool = True,
#         scale: float = 0.000305,
#         shift: float = 0.1378,
#         invert: bool = False,
#         lr: float = 0.0001,
#         weight_decay: float = 0.005,
#         loss_type: string = "eigen",
#     ):
#         super().__init__()

#         # this line allows to access init params with 'self.hparams' attribute
#         # it also ensures init params will be stored in ckpt
#         self.save_hyperparameters(logger=False)

#         self.model = DPTDepthModel(path, non_negative, scale, shift, invert)

#         # loss function
#         self.criterion = DepthLoss(loss_type)

#         # self.automatic_optimization = False
    
#     def forward(self, x: torch.Tensor):
#         return self.model(x)

#     def step(self, batch: Any):
#         in_, mask, gt = batch['image'], batch['mask'], batch['depth']
#         pred = self.forward(in_)
#         loss = self.criterion(pred, gt)
#         return loss, pred, gt

#     def training_step(self, batch: Any, batch_idx: int):
        
#         # opt = self.optimizers()
#         # opt.zero_grad()

#         loss, preds, targets = self.step(batch)
#         self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
#         # input_visual = rgb_unnormalize(batch['image'][0])
#         # preds_visual = depth_visualization(preds[0])
#         # gt_visual = depth_visualization(batch['depth'][0])
#         # tensor_logger = self.logger.experiment[0]
#         # tensor_logger.add_image(
#         #     'train/input_rgb', input_visual, self.global_step
#         # )
#         # tensor_logger.add_image(
#         #     'train/pred_depth', preds_visual, self.global_step
#         # )
#         # tensor_logger.add_image(
#         #     'train/gt_depth', gt_visual, self.global_step
#         # )

#         # we can return here dict with any tensors
#         # and then read it in some callback or in `training_epoch_end()`` below
#         # remember to always return loss from `training_step()` or else backpropagation will fail!
#         # self.manual_backward(loss)

#         return loss
        
#     def training_epoch_end(self, outputs: List[Any]):
#         # `outputs` is a list of dicts returned from `training_step()`
#         pass

#     def validation_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)
#         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
#         input_visual = rgb_unnormalize(batch['image'][0])
#         preds_visual = depth_visualization(preds[0])
#         gt_visual = depth_visualization(batch['depth'][0])
#         tensor_logger = self.logger.experiment[0]
#         tensor_logger.add_image(
#             f'val/input_rgb', input_visual, self.global_step
#         )
#         tensor_logger.add_image(
#             f'val/pred_depth', preds_visual, self.global_step
#         )
#         tensor_logger.add_image(
#             f'val/gt_depth', gt_visual, self.global_step
#         )

#         return loss

#     def validation_epoch_end(self, outputs: List[Any]):
#         # acc = self.val_acc.compute()  # get val accuracy from current epoch
#         # self.val_acc_best.update(acc)
#         # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
#         pass

#     def test_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)
#         self.log("test/loss", loss, on_step=False, on_epoch=True)

#         return loss

#     def test_epoch_end(self, outputs: List[Any]):
#         pass

#     def on_epoch_end(self):
#         # reset metrics at the end of every epoch
#         pass

#     def configure_optimizers(self):
#         """Choose what optimizers and learning-rate schedulers to use in your optimization.
#         Normally you'd need one. But in the case of GANs or similar you might have multiple.
#         See examples here:
#             https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
#         """
#         return torch.optim.Adam(
#             params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
#         )