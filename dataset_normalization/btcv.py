import os
import yaml

import numpy as np
import nibabel as nib
from monai.transforms import (
    LoadImaged, 
    Compose, 
    Spacingd, 
    EnsureChannelFirstd, 
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Invertd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    RandAffine,

)

from utils import load_from_file


class NormalizedDataset:

    def __init__(self) -> None:
        pass

    def __call__(self):
        pass

    def pre_process(self):
        return Compose(
                    [
                    Orientationd(keys=self.keys[0], axcodes="RAS"),
                    Spacingd(
                        keys=self.keys,
                        pixdim=(1.5, 1.5, 2.0),
                        mode=("nearest"),
                    ),
                    ScaleIntensityRanged(
                        keys=self.keys,
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    SaveImaged(keys=self.keys, output_dir="./results")
                
                    ]
                )




if __name__ == "__main__":
    # Lee el archivo YAML
    with open("paths.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    src_path = load_from_file(config['btcv']['3D']['src_path'])
    tgt_path = load_from_file(config["btcv"]["3D"]["dst_path"])

    print(src_path)

    


    path_name_list = os.listdir(src_path + "/imagesTr")



    name_list = [name.split(".")[0] for name in path_name_list]


    """
    if not os.path.exists(tgt_path+"/list"):
        os.makedirs("%s/list"%(tgt_path))
    with open("%s/list/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for name in name_list:
        img_name = name + ".nii.gz"
        lab_name = img_name.replace("img", "label")

        img = sitk.ReadImage(src_path+"/imagesTr/%s"%img_name)
        lab = sitk.ReadImage(src_path+"/labelsTr/%s"%lab_name)

        ResampleImage(img, lab, tgt_path, name, (1.5, 1.5, 2.0)) #  (0.767578125, 0.767578125, 1.0)  ||| (1.5, 1.5, 2.0)
        print(img_name, 'done')
        print(lab_name, 'done')




    # Load the images
    images = images.get_fdata()
    labels = labels.get_fdata()

    # Define the transformations
    transformations = Compose([
        LoadImaged(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["pixdim"], mode=("bilinear", "nearest")),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Invertd(keys=["image"], transform=ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)),
        Resized(keys=["image", "label"], spatial_size=config["spatial_size"]),
        SaveImaged(keys=["image", "label"], output_dir=config["output_dir"], output_postfix="normalized"),
    ])

    # Apply the transformations
    data = {"image": images, "label": labels}
    data = transformations(data)

    print("Data normalized and saved successfully.")    
    """