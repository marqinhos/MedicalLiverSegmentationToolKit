import subprocess
import time
import os
import glob
import json
from datetime import datetime
from multiprocessing import Process

from metrics.classMetrics import (
    RemovirtMetrics,
    MetricResult,
)

from metrics.saveMetrics import SaveMetricsJson



class SequentialMetrics:
    models_2d = [] # TODO
    models_3d = [
        #'attention_unet',
        'medformer', 
        'resunet', 
        'swin_unetr', 
        'unet++', 
        'unetr', 
        'vnet', 
        'segformer'
    ]

    dimensions = 3

    file_name = 'metrics.json'
    root_path = './results/'
    gt_img_path = "../../Datasets/BTCV_/imagesTs/"
    gt_lbl_path = "../../Datasets/BTCV_/labelsTs/"

    name_dataset = 'BTCV'

    all_metrics = MetricResult.metrics

    all_classes = [ '__BKG__','Spleen','Right Kidney','Left Kideny','Gallbladder',
                'Esophagus','Liver', 'Stomach','Aorta','IVC','Portal and Splenic Veins',
                'Pancreas','Right adrenal gland','Left adrenal gland']

    def __load_from_file(self, path):
        """Function to load the data from a file.

        Args:
            path (str): Path to the file.
        """
             
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), path))  
        return path

    def __call__(self):
        metrics = RemovirtMetrics(self.all_classes)
        save_metrics = SaveMetricsJson()

        
        for name_model in self.models_3d:
            network_path = self.__load_from_file(os.path.join(self.root_path, name_model))
            network_path += f"_{self.dimensions}d"
            if os.path.exists(network_path):
                for pred_filename in glob.glob(os.path.join(network_path, '*.nii.gz')):
                    base_name_pred = os.path.basename(pred_filename)  
                    name_pred_without_extension = os.path.splitext(base_name_pred)[0] 
                    gt_name = name_pred_without_extension.split("_Pred")[0]  # Result: img0061
                    lbl_name_with_extension = gt_name.replace("img", "label") + ".nii.gz"  # Result: label0061.nii.gz
                    img_name_with_extension = gt_name + ".nii.gz"
                    
                    results = metrics(
                        [self.gt_img_path+img_name_with_extension, 
                         self.gt_lbl_path+lbl_name_with_extension], 
                         "."+self.root_path+name_model+f"_{self.dimensions}d/"+base_name_pred
                        )
                    print("Saving {} metrics of {}".format(name_model, gt_name))
                    for name_metric in self.all_metrics:
                        save_metrics(
                            self.file_name,
                            name_pred_without_extension.split(".nii")[0],
                            gt_name,
                            self.dimensions,
                            self.name_dataset,
                            name_model,
                            name_metric,
                            self.all_classes,
                            results.results

                        )



if __name__ == "__main__":
    SequentialMetrics()()
