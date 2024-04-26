import os

import numpy as np
from typing import Union, List

import plotly.graph_objs as go
import nibabel as nib
from plotly.subplots import make_subplots
# from concurrent.futures import ThreadPoolExecutor # TODO more faster
from monai.data import Dataset
from monai.transforms import (
    LoadImaged, 
    Compose, 
    Spacingd, 
    EnsureChannelFirstd, 
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    )
from skimage import measure
from scipy.ndimage import (
    generate_binary_structure, 
    binary_erosion, 
    distance_transform_edt,
    binary_dilation
    )


class MetricResult:
    """ Class to store the results of the metrics. With this class, you can compute the mean of all classes and plot the results.
    """
    metrics = ["DSC", "NSD", "MASD", "HD", "RVD"]
    range_metrics = {"DSC": [0, 1],
                     "NSD": [0, 1],
                     "MASD": [0, np.inf],
                     "HD": [0, np.inf],
                     "RVD": [-1, 1]
    }


    def __init__(self, result):
        """Function to initialize the class.

        Args:
            result (dict): Dictionary with the results.
        """        
        self.results = result 


    def __str__(self):
        """Method to return the string representation of the object.

        Returns:
            str: String representation of the object
        """        
        result_str = ""
        for classes, metrics in self.results.items():
            result_str += f"{classes}:\n"
            for metric, value in metrics.items():
                result_str += f"  {metric}: {value}\n"
            result_str += "\n"
        return result_str


    def mean(self):
        """Function to compute the mean of all classes.

        Returns:
            dict: Mean of all classes
        """        
        
        return  {
                    metric: sum(self.results[classes][metric] for classes in self.results) / len(self.results)
                    for metric in self.results[list(self.results.keys())[0]]
                }


    def plot(self):
        """Function to plot the results.
        """        
        classes = list(self.results.keys())
        metrics = list(self.results[classes[0]].keys())

        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
              'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
              'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
              'rgb(23, 190, 207)']

        fig = make_subplots(rows=len(metrics), 
                            cols=1, 
                            subplot_titles=[
                                r"$\text{Range: }[0, 1]\uparrow  $ DSC",
                                r"$\text{Range: }[0, 1]\uparrow$ NSD",
                                r"$\text{Range: }[0, \infty]\downarrow$ MASD",
                                r"$\text{Range: }[0, \infty]\downarrow$ HD",
                                r"$\text{Range: }\uparrow[-1, 1]\downarrow$ RVD"
                            ], 
                            shared_xaxes=True)

        for i, metric in enumerate(metrics):
            values = [self.results[cls][metric] for cls in classes]
            std_values = [np.std(values) for _ in classes]

            error_y = dict(
                type='data',
                array=std_values,
                visible=True,
                color='rgba(31,119,180,0.7)'
            )

            fig.add_trace(
                go.Bar(x=classes, y=values, name=metric, marker_color='skyblue', error_y=error_y),
                row=i+1, col=1
            )
            
            fig.update_yaxes(title_text=metric, row=i+1, col=1)

        fig.update_xaxes(title_text="Classes", row=len(metrics)//4, col=1)

        fig.update_layout(showlegend=False, height=len(metrics)*200, title_text="Metrics per Class")
        fig.show()
    
    def get_dict(self):
        return self.result
    


class VolumeVisualization:
    """Class to visualize the 3D volumes.
    """
    color_scale=[
            [0.0, 'rgb(253, 225, 197)'],
            [0.1, 'rgb(253, 216, 179)'],
            [0.2, 'rgb(253, 207, 161)'],
            [0.3, 'rgb(253, 194, 140)'],
            [0.4, 'rgb(253, 181, 118)'],
            [0.5, 'rgb(253, 167, 97)'],
            [0.6, 'rgb(253, 153, 78)'],
            [0.7, 'rgb(252, 140, 59)'],
            [0.8, 'rgb(248, 126, 43)'],
            [0.9, 'rgb(243, 112, 27)'],
            [1.0, 'rgb(236, 98, 15)']]
        
    color_scale2=[
        [0.0, 'rgb(203, 225, 197)'],
        [0.1, 'rgb(203, 216, 179)'],
        [0.2, 'rgb(203, 207, 161)'],
        [0.3, 'rgb(203, 194, 140)'],
        [0.4, 'rgb(203, 181, 118)'],
        [0.5, 'rgb(203, 167, 97)'],
        [0.6, 'rgb(203, 153, 78)'],
        [0.7, 'rgb(202, 140, 59)'],
        [0.8, 'rgb(198, 126, 43)'],
        [0.9, 'rgb(193, 112, 27)'],
        [1.0, 'rgb(176, 98, 15)']]
    

    def __init__(self):
        pass
    
    def plot3D(self, vol_GT, vol_Pr, class_name, class_index, threshold=0, step_size=1):
        """Function to plot the 3D visualization of the ground truth and the prediction of a specific class in the prediction.

        Args:
            vol_GT (numpy.ndarray): Ground Truth
            vol_Pr (numpy.ndarray): Prediction
            class_name (str): Class name
            class_index (int): Class index
            threshold (int, optional): Threshold use to extract cubes from the volumes. Defaults to 0.
            step_size (int, optional): Detail for representation. Defaults to 1.
        """       

        gt = np.zeros_like(vol_GT, dtype = int)
        gt[vol_GT == class_index] = 1
        pred = np.zeros_like(vol_Pr, dtype = int)
        pred[vol_Pr == class_index] = 1

        verts_GT, faces_GT, _, _ = measure.marching_cubes(gt, threshold, step_size=step_size, allow_degenerate=True)
        x_GT, y_GT, z_GT = verts_GT.T
        I_GT, J_GT, K_GT = faces_GT.T
        verts_Pr, faces_Pr, _, _ = measure.marching_cubes(pred, threshold, step_size=step_size, allow_degenerate=True)
        x_Pr, y_Pr, z_Pr = verts_Pr.T
        I_Pr, J_Pr, K_Pr = faces_Pr.T

        mesh_GT = dict(
            x=x_GT,
            y=y_GT,
            z=z_GT,
            colorscale=self.color_scale,
            intensity=z_GT,
            i=I_GT,
            j=J_GT,
            k=K_GT,
            name="Ground Truth",
            opacity=1,
            showscale=False,
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                fresnel=0.1,
                specular=1,
                roughness=0.05,
                facenormalsepsilon=1e-8,
                vertexnormalsepsilon=1e-15
            ),
            lightposition=dict(x=100, y=200, z=0)
        )
        mesh_Pr = dict(
            x=x_Pr,
            y=y_Pr,
            z=z_Pr,
            colorscale=self.color_scale2,
            intensity=z_Pr,
            i=I_Pr,
            j=J_Pr,
            k=K_Pr,
            name="Prediction",
            opacity=1,
            showscale=False,
            lighting=dict(
                ambient=0.18,
                diffuse=1,
                fresnel=0.1,
                specular=1,
                roughness=0.05,
                facenormalsepsilon=1e-8,
                vertexnormalsepsilon=1e-15
            ),
            lightposition=dict(x=100, y=200, z=0)
        )

        axis_style = dict(showbackground=True,
                        backgroundcolor="rgb(230, 230,230)",
                        gridcolor="rgb(255, 255, 255)",
                        zerolinecolor="rgb(255, 255, 255)")

        title = "Visualizations of the Ground Truth and the Prediction of the class {class_name}".format(class_name=class_name)
        layout = dict(
            title=title,
            font=dict(family='Balto'),
            scene=dict(
                camera=dict(eye=dict(x=1.15, y=-1.15, z=0.9)),
                aspectratio=dict(x=1, y=1., z=0.8),
                xaxis=dict(axis_style),
                yaxis=dict(axis_style),
                zaxis=dict(axis_style)
            )
        )

        figure = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'mesh3d'},
                                    {'type': 'mesh3d'}]],
                            subplot_titles=['Ground Truth of the {class_name} class'.format(class_name=class_name), 'Prediction of the {class_name} class'.format(class_name=class_name)])
        chunk = go.Mesh3d(mesh_GT)
        figure.add_trace(chunk, row=1, col=1)
        chunk = go.Mesh3d(mesh_Pr)
        figure.add_trace(chunk, row=1, col=2)
        figure.update_layout(layout)
        figure.show()
    


class ValidateMetrics:
    """Class to validate the metrics.
    """
    def create_sphere_mask(self, shape, radius):
        """Function to create a sphere mask.

        Args:
            shape (numpy.ndarray): Shape of the mask
            radius (int): Radius of the sphere

        Returns:
            numpy.ndarray: Sphere mask as a numpy array
        """       

        grid = np.ogrid[[slice(-dim // 2, dim // 2) for dim in shape]]

        distances = [coord**2 for coord in grid]
        distance_from_center = np.sqrt(sum(distances))

        sphere_mask = (distance_from_center <= radius).astype(np.uint8) # Create a sphere mask

        return sphere_mask


    def create_ellipsoid_mask(self, shape, radii):
        """Function to create an ellipsoid mask.

        Args:
            shape (numpy.ndarray): Shape of the mask
            radii (numpy.ndarray): Radii of the ellipsoid. For vertical, horizontal and depth.

        Returns:
            numpy.ndarray: Ellipsoid mask as a numpy array
        """ 

        grid = np.ogrid[[slice(-dim // 2, dim // 2) for dim in shape]]

        coords = [(coord / radius) ** 2 for coord, radius in zip(grid, radii)]
        distances = sum(coords)

        ellipsoid_mask = (distances <= 1).astype(np.uint8) # Create a ellipsoid mask

        return ellipsoid_mask



class RemovirtMetrics(VolumeVisualization):
    """Class to compute the metrics for the segmentation of the organs.
    Metrics available to compute:
        Dice Similarity Coefficient (DSC)
        Normalize Surface Distance (NSD)
        Mean Average Surface Distance (MASD)
        Hausdorff Distance (HD)
        Relative Volume Difference (RVD)
    """

    def __init__(self, classes):
        """Constructor of the class.
        """       
        self.__min_dist_gt_pred = None
        self.__min_dist_pred_gt = None
        self.__s = None
        self.__s_prime = None
        
        self.keys_gt = ["image", "label"]
        self.keys_pred = ["prediction"]
        self.classes = classes
        

    def __call__(self, gt, pred, visualize_class_3d=None, spec_class=None, to_check=False):
        """Function to compute the metrics.
            : Union[str, np.ndarray, List[np.ndarray, TODO List[List[str]]]
         Args:
            path_gt (List[str] | np.ndarray, List[np.ndarray] | List[List[str]]): Path to the ground truth or the ground truth itself.
            path_pred (str | np.ndarray, List[np.ndarray] | List[str]): Path to the prediction or the prediction itself.
            visualize_class_3d (str, optional): Class to visualize in 3D. Defaults to None. Se the class name to visualize in 3D.
            spec_class (int, optional): Specific class to take metrics. Defaults to None.
            to_check (bool, optional): Only use if you want check the metrics. Defaults to False.

        Raises:
            ValueError: If the input is not valid.

        Returns:
            dict: Dictionary with all metrics
        """     

        gt, pred = self.get_gt_pred(gt, pred)

        if visualize_class_3d is not None:
            try:
                print("Show Volume of {visualize_class_3d} class in 3D.".format(visualize_class_3d=visualize_class_3d))
                self.plot3D(gt, pred, visualize_class_3d, self.classes.index(visualize_class_3d))    

            except Exception as e:
                print("Error: ", e)
                print("The class to visualize in 3D is not valid.") 

        print("Computing metrics...")

        return self.compute_all_classes_detect(gt, pred, spec_class, to_check)
   

    def get_gt_pred(self, gt, pred):
        """Function to get the ground truth and the prediction.

        Args:
            gt (List[str] | numpy.ndarray | TODO List[numpy.ndarray] | TODO List[List[str]]): Ground Truth
            pred (str | numpy.ndarray | TODO List[numpy.ndarray] | TODO List[str]]): Prediction

        Returns:
            np.ndarray: The ground truth and the prediction
        """        
        if isinstance(gt, list) and all(isinstance(item, str) for item in gt):
            data_img = self.__load_from_file(gt[0])
            data_label = self.__load_from_file(gt[1])

            data_dict = [{self.keys_gt[0]: img, self.keys_gt[1]: lbl} for img, lbl in zip(data_img, data_label)]
            load_image = LoadImaged(keys=self.keys_gt)
            dataset = Dataset(
                            data=data_dict, 
                            transform=load_image)
            gt_tensor = dataset[0][self.keys_gt[1]]
            gt = gt_tensor.squeeze().get_array()
            
        elif isinstance(gt, np.ndarray):
            gt = gt
        elif isinstance(gt, list) and all(isinstance(item, np.ndarray) for item in gt):
            # TODO
            raise ValueError("In development.")
        else:
            raise ValueError("The input for the ground truth is not valid.")

        if isinstance(pred, str):  
            data = self.__load_from_file(pred)
            data_dict = [{self.keys_pred[0]: img} for img in data]
            load_image = LoadImaged(keys=self.keys_pred)
            dataset = Dataset(
                            data=data_dict, 
                            transform=load_image)
            pred_tensor = dataset[0][self.keys_pred[0]]
            pred = pred_tensor.get_array()
            
        elif isinstance(pred, np.ndarray):
            pred = pred
        elif isinstance(pred, list) and all(isinstance(item, np.ndarray) for item in pred):
            # TODO
            raise ValueError("In development.")
        else:
            raise ValueError("The input for the prediction is not valid.")
        
        return self.__array2nifty(gt)[0].astype(int), self.__array2nifty(pred)[0].astype(int)  


    def compute_all_classes_detect(self, gt_all_classes, pred_all_classes, spec_class=None, to_check=False):
        """Function to compute the metrics for all classes or a specific class in the prediction.

        Args:
            gt_all_classes (_type_): Ground Truth for all classes
            pred_all_classes (_type_): Predictions  for all classes
            spec_class (_type_, optional): Specific class to take metrics. Defaults to None.
            to_check (bool, optional): Only use if you want check the metrics. Defaults to False.

        Returns:
            dict: Metrics for all classes or a specific class
        """   
        n_classes_Pr = np.unique(pred_all_classes)
        result = {}
        if spec_class is None and not to_check:
            for class_detect in n_classes_Pr:
                gt = np.zeros_like(gt_all_classes, dtype = int)
                gt[gt_all_classes == class_detect] = 1
                pred = np.zeros_like(pred_all_classes, dtype = int)
                pred[pred_all_classes == class_detect] = 1
                result[self.classes[class_detect]] = {
                        "DSC": self.dsc(gt, pred),
                        "NSD": self.nsd(gt, pred),
                        "MASD": self.masd(gt, pred),
                        "HD": self.hd(gt, pred),
                        "RVD": self.rvd(gt, pred)
                    }

                self.__min_dist_gt_pred = None # To reset for next MASD and HD measures
            return MetricResult(result)
        
        elif not to_check:
            gt = np.zeros_like(gt_all_classes, dtype = int)
            gt[gt_all_classes == spec_class] = 1
            pred = np.zeros_like(pred_all_classes, dtype = int)
            pred[pred_all_classes == spec_class] = 1
            return MetricResult({
                "DSC": self.dsc(gt, pred),
                "NSD": self.nsd(gt, pred),
                "MASD": self.masd(gt, pred),
                "HD": self.hd(gt, pred),
                "RVD": self.rvd(gt, pred)
            })
        
        else:
            gt = gt_all_classes
            pred = pred_all_classes
            return MetricResult({
                "DSC": self.dsc(gt, pred),
                "NSD": self.nsd(gt, pred),
                "MASD": self.masd(gt, pred),
                "HD": self.hd(gt, pred),
                "RVD": self.rvd(gt, pred)
            })


    def dsc(self, gt, pred):
        """Function to get the Dice Similarity Coefficient (DSC) metric.

        Args:
            gt (numpy.ndarray |): Ground Truth
            pred (numpy.ndarray |): Prediction

        Returns:
            float: The DSC metric
        """      
        if np.sum(gt) == 0: return None

        return (2 * np.abs(np.sum(gt * pred))) \
                / (np.abs(np.sum(gt)) + np.abs(np.sum(pred))) if np.sum(gt) != 0 else None
    

    def rvd(self, gt, pred):
        """Function to get the Relative Volume Difference (RVD) metric.

        Args:
            gt (numpy.ndarray | ): Ground Truth
            pred (numpy.ndarray | ): Prediction
        
        Returns:
            float: The RVD metric
        """
        if np.sum(gt) == 0: return None

        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        return (np.sum(gt) - np.sum(pred)) / np.sum(gt) if np.sum(gt) != 0 else None


    def masd(self, gt, pred):
        """Function to get the Mean Average Surface Distance (MASD) metric.

        Args:
            gt (numpy.ndarray): Ground Truth
            pred (numpy.ndarray): Prediction

        Returns:
            float: The MASD metric
        """ 
        if np.sum(gt) == 0: return None
        
        if self.__min_dist_gt_pred is None:
            self.__set_aux_2_masd_hd(gt, pred)
        
        sds_A_to_B = np.ravel(self.__min_dist_gt_pred[self.__s_prime != 0])
        sds_B_to_A = np.ravel(self.__min_dist_pred_gt[self.__s != 0])

        
        return (0.5 * ((sds_A_to_B.sum() / len(sds_A_to_B)) 
                        + (sds_B_to_A.sum() / len(sds_B_to_A)))) if len(sds_A_to_B) != 0 and len(sds_B_to_A) != 0 else None


    def hd(self, gt, pred):
        """Function to get the Hausdorff Distance (HD) metric.

        Args:
            gt (numpy.ndarray): Ground Truth
            pred (numpy.ndarray): Prediction

        Returns:
            float: The HD metric
          
        """
        if np.sum(gt) == 0: return None

        if self.__min_dist_gt_pred is None:
            self.__set_aux_2_masd_hd(gt, pred)
        
        return np.concatenate([
            np.ravel(self.__min_dist_gt_pred[self.__s_prime != 0]), 
            np.ravel(self.__min_dist_pred_gt[self.__s != 0])]).max() if np.sum(gt) != 0 else None
    

    def nsd(self, gt, pred, tau=2):  
        """Function to get the Normalize Surface Distance (NSD) metric. 
        The NSD is a measure of the average distance between the surfaces of the ground truth and the prediction.
    

        Args:
            gt (numpy.ndarray): Ground Truth
            pred (numpy.ndarray): Prediction
            tau (int, optional): Tolerance parameter to represent the degree of strictness for what constitutes a correct boundary. Defaults to 2.

        Returns:
            float: The NSD metric
        """  
        if np.sum(gt) == 0: return None
        
        mask_true_boundary = self.__get_boundary(gt)
        mask_true_border_region = self.__get_boundary(mask=gt, 
                                                      border_region=True, 
                                                      tau=tau)

        mask_pred_boundary = self.__get_boundary(pred)
        mask_pred_border_region = self.__get_boundary(mask=pred, 
                                                      border_region=True, 
                                                      tau=tau)

        intersection_true_pred = np.sum(mask_true_boundary * mask_pred_border_region)
        intersection_pred_true =  np.sum(mask_pred_boundary * mask_true_border_region)
        smooth = 1e-7
        return ((np.abs(intersection_true_pred) + np.abs(intersection_pred_true) + smooth) 
               / (np.abs(np.sum(mask_true_boundary)) + np.abs(np.sum(mask_pred_boundary)) + smooth))


    def __load_from_file(self, path):
        """Function to load the data from a file.

        Args:
            path (str): Path to the file.
        """
             
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), path))  
        return [path]

    def __set_aux_2_masd_hd(self, gt, pred, sampling=1, connectivity=1):
        """Auxiliary function to set the auxiliary variables for MASD and HD metrics.

        Args:
            gt (numpy.ndarray): Ground Truth
            pred (numpy.ndarray): Prediction
            sampling (int, optional): T. Defaults to 1.
            connectivity (int, optional): . Defaults to 1.
        """        
        
        input_1 = np.atleast_1d(gt.astype(np.bool_))
        input_2 = np.atleast_1d(pred.astype(np.bool_))
        
        conn = generate_binary_structure(input_1.ndim, connectivity)

        self.__s = np.bitwise_xor(input_1,binary_erosion(input_1, conn))
        self.__s_prime = np.bitwise_xor(input_2, binary_erosion(input_2, conn))
        
        self.__min_dist_gt_pred = distance_transform_edt(~self.__s, sampling)
        self.__min_dist_pred_gt = distance_transform_edt(~self.__s_prime, sampling)

    def __get_boundary(self, mask, border_region=False,tau=2):
        """Auxiliary function to obtain the boundary of a mask.

        Args:
            mask (numpy.ndarray): Mask
            border_region (bool, optional): If True, the boundary is dilated. Defaults to False.
            tau (int, optional): Tolerance parameter to represent the degree of strictness for what constitutes a correct boundary. Defaults to 2.

        Returns:
            numpy.ndarray: The boundary of the mask
        """        
        if border_region:         
            dilated = binary_dilation(mask, iterations=tau)
            eroded = binary_erosion(mask, iterations=tau)
            return np.logical_xor(dilated, eroded).astype(np.uint8)
        else:
            dilated = binary_dilation(mask)
            return (dilated - mask).astype(np.uint8)
    
    def __array2nifty(self, array):
        img = nib.Nifti1Image(array, affine=np.eye(4))
        return img.get_fdata(), img.header.get_zooms()


if "__main__" == __name__:

    # Test the class
    # Example de uso
    shape = (234, 199, 206)
    center = np.array([32, 32, 32])  # Sphere center
    radii = np.array([20, 25, 15]) # Slips radius
    radius = 20

    validator = ValidateMetrics()

    mask_true = validator.create_sphere_mask(shape, radius)
    mask_pred = validator.create_ellipsoid_mask(shape, radii)  

    # Paths
    gt_img_path = "../../Datasets/BTCV_/imagesTs/img0061.nii.gz" 
    gt_lbl_path = "../../Datasets/BTCV_/labelsTs/label0061.nii.gz"
    pred_path = "../results/attention_unet_3d/img0061_Pred.nii.gz"

    # Classes
    organs = [ '__BKG__','Spleen','Right Kidney','Left Kideny','Gallbladder',
                'Esophagus','Liver', 'Stomach','Aorta','IVC','Portal and Splenic Veins',
                'Pancreas','Right adrenal gland','Left adrenal gland']


    metrics = RemovirtMetrics(organs)

    results = metrics([gt_img_path, gt_lbl_path], pred_path) # To show liver on 3D. Mean of all classes.
    #results = metrics([gt_img_path, gt_lbl_path], pred_path, spec_class=6) # Specific class to take metrics
    # results = metrics(mask_true, mask_true, to_check=True) # To check the metrics
    print(results)
    print(results.results)
    # print(results.mean())
    #results.plot()
    
    
    print("End of the test.")

   