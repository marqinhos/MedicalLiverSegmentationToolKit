import os
import yaml

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from monai.data import Dataset
from monai.transforms import (
    LoadImaged, 
    Compose, 
    Spacingd, 
    EnsureChannelFirstd, 
    Orientationd,
    ScaleIntensityRanged,
)
from sklearn.model_selection import train_test_split

from utils import load_from_file


class NormalizedDataset:
    """Class to normalize a dataset
    """    

    def __init__(self):
        """Constructor of the NormalizedDataset class.
        """        
        self.path_img_list = {}
        self.path_label_list = {}
        self.img_names_list = None
        self.keys = ["image", "label"]


    def __call__(self, src_path, tgt_path, train_percentage):
        """Function to normalize the dataset and save it in the target path.

        Args:
            src_path (str): Full path to the source dataset.
            tgt_path (str): Full path to the target dataset.
            train_percentage (float): Percentage of the dataset to be used for training.
        """        
        name_list_ext_tr = os.listdir(src_path + "/imagesTr")
        name_list_ext_ts = os.listdir(src_path + "/imagesTs")

        self.img_names_list = name_list_ext_tr + name_list_ext_ts

        for path_img_tr in name_list_ext_tr:
            self.path_img_list[path_img_tr] = src_path+"/imagesTr/"+path_img_tr
            path_lbl_tr = path_img_tr.replace("img", "label")
            self.path_label_list[path_lbl_tr] = src_path+"/labelsTr/"+ path_lbl_tr

        for path_img_ts in name_list_ext_ts:
            self.path_img_list[path_img_ts] = src_path+"/imagesTs/"+path_img_ts 
            path_lbl_ts = path_img_ts.replace("img", "label")
            self.path_label_list[path_lbl_ts] = src_path+"/labelsTs/"+ path_lbl_ts

        test_img_names, train_img_names = self.split_dataset(self.img_names_list, train_percentage)
        
        name_list = [name.split('.')[0] for name in train_img_names]

        if not os.path.exists(tgt_path+"/list"):
            os.makedirs("%s/list"%(tgt_path))
        with open("%s/list/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
            yaml.dump(name_list, f)

        self.__apply_transformations(train_img_names, tgt_path, "")
        

    def transformations(self, pixdim):
        """Function to apply the transformations to the dataset.

        Args:
            pixdim (tuple): Tuple with the pixel dimensions.

        Returns:
            monai.transforms.compose.Compose: Compose object with the transformations to be applied.
        """        
        return Compose(
                    [
                    LoadImaged(keys=self.keys, image_only=True),
                    EnsureChannelFirstd(keys=self.keys),
                    Orientationd(keys=self.keys, axcodes="RAS"),
                    Spacingd(
                        keys=self.keys,
                        pixdim=(1.5, 1.5, 2), #
                        mode=("nearest"),
                    ),
                    ScaleIntensityRanged(
                        keys=self.keys[0],
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    ]
                )


    def split_dataset(self, name_list_ext, test_percentage):
        """Function to split the dataset into training and testing.

        Args:
            name_list_ext (list): List with the names of the images.    
            test_percentage (float): Percentage of the dataset to split it.

        Returns:
            list, list: Lists with the names of the images for training and testing. 
        """        
        return train_test_split(name_list_ext, test_size=test_percentage, random_state=42)
    

    def __apply_transformations(self, data, tgt_path):
        """Function to apply the transformations to the dataset.

        Args:
            data (list): List with the data to be transformed.
            tgt_path (str): Full path to the target dataset.
        """        
        for img_name in data:
            lbl_name = img_name.replace("img", "label")

            img = nib.load(self.path_img_list[img_name])
            lbl = nib.load(self.path_label_list[lbl_name])

            print(img.get_fdata().shape)
            #data = {self.keys[0]: img.get_fdata(), self.keys[1]: lbl.get_fdata()}
            transformation = self.transformations(img.header.get_zooms())
            data_dict = [{self.keys[0]: self.path_img_list[img_name], self.keys[1]: self.path_label_list[lbl_name]}]
            ## Important data_dict be a list of dictionaries
            dataset_0 = Dataset(data=data_dict, transform=transformation)

            img_normalized = dataset_0[0][self.keys[0]].squeeze().get_array()
            lbl_normalized = dataset_0[0][self.keys[1]].squeeze().get_array()
            print(img_normalized.shape)
            
            img_name = img_name.replace(".nii.gz", "")
            lbl_name = lbl_name.replace(".nii.gz", "")

            sitk_img = sitk.GetImageFromArray(img_normalized)
            sitk_lbl = sitk.GetImageFromArray(lbl_normalized)

            sitk.WriteImage(sitk_img, '%s/%s.nii.gz'%(tgt_path, img_name))
            sitk.WriteImage(sitk_lbl, '%s/%s_gt.nii.gz'%(tgt_path, img_name))
            print(img_name, 'done')
            print(lbl_name, 'done')

        

if __name__ == "__main__":
    with open("paths.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    src_path = load_from_file(config['btcv']['3D']['src_path'])
    tgt_path = load_from_file(config["btcv"]["3D"]["dst_path"])
    train_percentage = float(config["btcv"]["3D"]["train_percentage"])

    dataset = NormalizedDataset()
    dataset(src_path, tgt_path, train_percentage)

    print("Data normalized and saved successfully.")    
    