import os


import torch
from torch.utils.data import Dataset as torchDataset
from monai.data import (
    Dataset as monaiDataset,
    CacheDataset,
    )
import numpy as np
import yaml
import SimpleITK as sitk
import nibabel as nib
import argparse

from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    # EnsureTyped,
    EnsureType,
)



class BTCVDataset(torchDataset):
    
    def __init__(self, args, mode="train", k_fold=5, k=0, seed=0):
        self.keys = ["image", "label"]
        self.spatial_size = (86, 86, 86) # be careful with z is 88 for 3D
        self.factor_increase_dataset = 100  

        self.mode = mode
        self.args = args
        self.path = self.load_from_file(args.data_root)

        assert mode in ['train', 'test'], "mode must be either 'train' or 'test'"

        with open(os.path.join(self.path, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list = yaml.load(f, Loader=yaml.SafeLoader)

        print('Start loading %s data'%self.mode)

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []

        for name in img_name_list:
            img_name = name + '.nii.gz'
            lbl_name = name + '_gt.nii.gz'

            img = nib.load(os.path.join(self.path, img_name))
            spacing = img.header.get_zooms()

            self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

            #img, lab = self.preprocess(img_name, lbl_name)

            self.img_list.append(os.path.join(self.path, img_name))
            self.lab_list.append(os.path.join(self.path, lbl_name))
            



    def __len__(self):
        if self.mode == 'train':
            return len(self.img_list) * self.factor_increase_dataset
        else:
            return len(self.img_list)


    def __getitem__(self, idx):
        idx = idx % len(self.img_list)

        img_path = self.img_list[idx]
        lbl_path = self.lab_list[idx]

        if self.mode == 'train':
            data_dict = [{self.keys[0]: img_path, self.keys[1]: lbl_path}]
            dataset = CacheDataset(
                    data=data_dict, 
                    transform=self.get_augmentation_transform()
                )
            
            tensor_img = dataset[0][0][self.keys[0]].as_tensor()
            tensor_lbl = dataset[0][0][self.keys[1]].as_tensor()

        else:
            pass

        
        assert tensor_img.shape == tensor_lbl.shape

        if self.mode == 'train':
            return tensor_img, tensor_lbl
        else:
            return tensor_img, tensor_lbl, np.array(self.spacing_list[idx])

    
    def load_from_file(self, path):
        """Function to load the data from a file.

        Args:
            path (str): Path to the file.
        """
                
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), path))  
        return path

    
    def preprocess(self, img_name, lbl_name):
        pass
        return self.load_img(img_name), self.load_img(lbl_name)


    def get_augmentation_transform(self): 
        train_transforms = Compose([
            LoadImaged(keys=self.keys, image_only=True),
            EnsureChannelFirstd(keys=self.keys),
            CropForegroundd(
                allow_smaller=False,
                keys=self.keys, 
                source_key=self.keys[0]
                ),
            RandCropByPosNegLabeld(
                    keys=self.keys,
                    label_key=self.keys[1],
                    spatial_size=self.spatial_size,
                    pos=1,
                    neg=1,
                    num_samples=1, #4
                    image_key=self.keys[0],
                    image_threshold=0,
                ),
            RandFlipd(
                    keys=self.keys,
                    spatial_axis=[0],
                    prob=0.10,
                ),
            RandFlipd(
                    keys=self.keys,
                    spatial_axis=[1],
                    prob=0.10,
                ),
            RandFlipd(
                    keys=self.keys,
                    spatial_axis=[2],
                    prob=0.10,
                ),
            RandRotate90d(
                    keys=self.keys,
                    prob=0.10,
                    max_k=3,
                ),
            RandShiftIntensityd(
                    keys=self.keys[0],
                    offsets=0.10,
                    prob=0.50,
                ),
            ])
        return train_transforms
 


if __name__ == "__main__":
    
    def get_parser():
        parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
        parser.add_argument('--dataset', type=str, default='btcv', help='dataset name')
        parser.add_argument('--model', type=str, default='medformer', help='model name')
        parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
        parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
        parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
        parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')
        parser.add_argument('--data_root', type=str, default='../../../Datasets/BTCV_normalized', help='unique experiment name')

        parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
        parser.add_argument('--load', type=str, default=False, help='load pretrained model')
        parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
        parser.add_argument('--log_path', type=str, default='./log/', help='log path')
        parser.add_argument('--unique_name', type=str, default='train', help='unique experiment name')
        
        parser.add_argument('--gpu', type=str, default='0')

        args = parser.parse_args()

        

        return args   

    args = get_parser()
    dataset = BTCVDataset(args)
    print(len(dataset))
    print("Data normalized and saved successfully.")    
    