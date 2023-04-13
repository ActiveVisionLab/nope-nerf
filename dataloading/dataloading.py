import os
import glob
import random
import logging
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from .dataset import DataField
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
logger = logging.getLogger(__name__)

def get_dataloader(cfg, mode='train',
                   shuffle=True, n_views=None):
    ''' Return dataloader instance

    Instansiate dataset class and dataloader and 
    return dataloader
    
    Args:
        cfg (dict): imported config for dataloading
        mode (str): tran/eval/render/all
        shuffle (bool): as name
        n_views (int): specify number of views during rendering
    '''
        
    batch_size = cfg['dataloading']['batchsize']
    n_workers = cfg['dataloading']['n_workers']
   
    fields = get_data_fields(cfg, mode)
    if n_views is not None and mode=='render':
        n_views = n_views
    else:
        n_views = fields['img'].N_imgs
    ## get dataset
    dataset = OurDataset(
         fields, n_views=n_views, mode=mode)

    ## dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, 
        shuffle=shuffle, pin_memory=True
    )

    return dataloader, fields


def get_data_fields(cfg, mode='train'):
    ''' Returns the data fields.

    Args:
        cfg (dict): imported yaml config
        mode (str): the mode which is used

    Return:
        field (dict): datafield
    '''
    use_DPT = (cfg['depth']['type']=='DPT')
    resize_img_transform = ResizeImage_mvs() # for dpt input images
    fields = {}
    load_ref_img = ((cfg['training']['pc_weight']!=0.0) or (cfg['training']['rgb_s_weight']!=0.0))
    dataset_name = cfg['dataloading']['dataset_name']
    if dataset_name=='any':
        img_field = DataField( 
                model_path=cfg['dataloading']['path'],
                transform=resize_img_transform,
                with_camera=True,
                with_depth=cfg['dataloading']['with_depth'],
                scene_name=cfg['dataloading']['scene'],
                use_DPT=use_DPT, mode=mode,spherify=cfg['dataloading']['spherify'], 
                load_ref_img=load_ref_img, customized_poses=cfg['dataloading']['customized_poses'],
                customized_focal=cfg['dataloading']['customized_focal'],
                resize_factor=cfg['dataloading']['resize_factor'], depth_net=cfg['dataloading']['depth_net'], 
                crop_size=cfg['dataloading']['crop_size'], random_ref=cfg['dataloading']['random_ref'], norm_depth=cfg['dataloading']['norm_depth'],
                load_colmap_poses=cfg['dataloading']['load_colmap_poses'], sample_rate=cfg['dataloading']['sample_rate'])
    else:
        print(dataset_name, 'does not exist')
    fields['img'] = img_field
    return fields
class ResizeImage_mvs(object):
    def __init__(self):
        net_w = net_h = 384
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal"
                    ),
                    normalization,
                    PrepareForNet(),
                ]
            )
    def __call__(self, img):
        img = self.transform(img)
        return img




class OurDataset(data.Dataset):
    '''Dataset class
    '''

    def __init__(self,  fields, n_views=0, mode='train'):
        # Attributes
        self.fields = fields
        print(mode,': ', n_views, ' views') 
        self.n_views = n_views

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.n_views

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data = {}
        for field_name, field in self.fields.items():
            field_data = field.load(idx)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        return data



def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
