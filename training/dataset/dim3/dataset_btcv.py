import os
import glob

import numpy as np
import nibabel as nib
import torch
import pytorch_lightning as pl

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)

from monai.transforms import (
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
)


class BTCVDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.keys = ["image", "label"]
        self.mode = ("nearest") #"bilinear", "nearest"
        self.spatial_size = args.roi_size
        self.pixdim = [1.5, 1.5, 2.0]
        self.scale_range = [-175, 250]

        self.train_files = [] 
        self.val_files = []
        self.test_files = []

        self.preprocess = None
        self.transform = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.my_dsc = None



    def prepare_data(self):

        data_images, data_labels = self.__get_data(folders_img_lbl=self.args.folders_img_lbl)

        if (len(data_images) != len(data_labels)) or len(data_images) ==0 or len(data_labels) ==0:
            raise TypeError("Error: Number of images and labels do not match or are empty")

        data_dicts = [
            {self.keys[0]: image_name, self.keys[1]: label_name}
            for image_name, label_name in zip(data_images, data_labels)
        ]

        min_pixdim = self.__get_less_pixdim(data_images)
        self.pixdim = [min_pixdim, min_pixdim, min_pixdim]

        
        percentage_val = round((1.0-self.args.percentage_train)/2, 2)
        train_num = int(np.floor(self.args.percentage_train * len(data_images)))
        val_num = int(np.floor(percentage_val * len(data_images)))
        test_num = len(data_images) - train_num - val_num
        assert (train_num + val_num + test_num) == len(data_images), "Trainers, val and test partition do not match the total"
        
        print("Total images {}: images for train {}, for val {} and for test {}".format(len(data_images),train_num, val_num, test_num))

        indices = np.random.permutation(len(data_images))
        indices_train = indices[:train_num]
        indices_val = indices[train_num:train_num+val_num]
        indices_test = indices[train_num+val_num:]

        self.train_files = [data_dicts[i] for i in indices_train]
        self.val_files   = [data_dicts[i] for i in indices_val] 
        self.test_files  = [data_dicts[i] for i in indices_test]
    

    def load_images_prediction(self):
       data_images = self.__get_data_pred(folders_img_lbl=self.args.folders_img_lbl)
       data_dicts = [
            {self.keys[0]: image_name}
            for image_name in data_images
        ]
       return data_dicts
    

    def get_preprocessing_transform(self):
        pre_transforms_0 = Compose(
            [
            EnsureChannelFirstd(keys=self.keys),
            Orientationd(keys=self.keys, axcodes="RAS"),
            Spacingd(
                keys=self.keys,
                pixdim=self.pixdim,
                mode=self.mode, 
            ),
            ScaleIntensityRanged(
                keys=self.keys,
                a_min=self.scale_range[0],
                a_max=self.scale_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                allow_smaller=False, 
                keys=self.keys,
                source_key=self.keys[0],
                ),
            ]
        )

        pre_transforms = Compose(
            [
            LoadImaged(keys=self.keys, image_only=True),
            pre_transforms_0
            ]
        )
        return pre_transforms, pre_transforms_0


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
                    num_samples=4,
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
                    keys=self.keys,
                    offsets=0.10,
                    prob=0.50,
                ),
            ])
        return train_transforms


    def get_preprocessing_transform_pred(self):
        val_test_transforms_0 = Compose(
            [
                EnsureChannelFirstd(keys=self.keys[0]),
                Orientationd(keys=self.keys[0], axcodes="RAS"),
                Spacingd(
                    keys=self.keys[0],
                    pixdim=self.pixdim,
                    mode=self.mode,
                ),
                ScaleIntensityRanged(
                    keys=self.keys[0],
                    a_min=self.scale_range[0],
                    a_max=self.scale_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(
                    allow_smaller=False, 
                    keys=self.keys[0], 
                    source_key=self.keys[0]
                    ),
                
            ]
        )

        val_test_transforms = Compose(
            [
                LoadImaged(keys=self.keys[0], image_only=True),
                val_test_transforms_0
            ]
        )

        return val_test_transforms, val_test_transforms_0


    def setup(self, stage=None):
        self.preprocess = self.get_preprocessing_transform()
        self.augment = self.get_augmentation_transform()

        if stage == "fit" or stage is None:
            self.train_ds = CacheDataset(
                data=self.train_files,
                transform=self.augment,
                cache_rate=self.args.cache_rate,
                num_workers=self.args.num_workers,
            )

            self.val_ds = CacheDataset(
                data=self.val_files,
                transform=self.preprocess,
                cache_rate=self.args.cache_rate,
                num_workers=self.args.num_workers,
            )
        
        if stage == "test" or stage is None:

            self.test_ds = CacheDataset(
                data=self.test_files,
                transform=self.preprocess,
                cache_rate=self.args.cache_rate,
                num_workers=self.args.num_workers,
            )


    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=list_data_collate,
        )
        return train_loader


    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=self.args.pin_memory, 
            collate_fn=list_data_collate,
        )
        return val_loader


    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=self.args.pin_memory, 
            collate_fn=list_data_collate,
        )
        return test_loader


    def __get_data(self, folders_img_lbl=True):

        if folders_img_lbl:
            data_images = sorted(
                glob.glob(os.path.join(self.args.data_dir, "imagesTr", "*.nii.gz")))
            data_labels = sorted(
                glob.glob(os.path.join(self.args.data_dir, "labelsTr", "*.nii.gz")))
            
            return data_images, data_labels

            
        else:
            all_files = sorted(
                glob.glob(os.path.join(self.args.data_dir, "*.nii.gz")))

            data_images = sorted(
                filter(lambda x: "_gt" in x, all_files))
            data_labels = sorted(
                filter(lambda x: "_gt" not in x, all_files))
            
            return data_images, data_labels
        

    def __get_data_pred(self, folders_img_lbl=True):

        if folders_img_lbl:
            data_images = sorted(
                glob.glob(os.path.join(self.args.data_dir, "imagesTs", "*.nii.gz")))
            
            return data_images

            
        else:
            all_files = sorted(
                glob.glob(os.path.join(self.args.data_dir, "*.nii.gz")))

            data_images = sorted(
                filter(lambda x: "_gt" in x, all_files))
            
            return data_images


    def __get_less_pixdim(self, data):
        pixdim = [nib.load(path).header.get_zooms() for path in data]
        pixdim = np.array(pixdim)
        return min(pixdim.min(axis=0))
