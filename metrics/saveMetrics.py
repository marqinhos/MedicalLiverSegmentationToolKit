import json
import os

class SaveMetricsJson:
    
    def __call__(
            self,
            filename, 
            name_pred, 
            name_gt, 
            dimensions,
            name_dataset, 
            name_model, 
            name_metric, 
            classes, 
            values_metrics
        ):
    
        previous_data = {}
        skeleton_data = self.__get_skeleton_data(
            name_pred, 
            name_gt, 
            name_dataset, 
            name_model, 
            name_metric, 
            dimensions
            )

        metric_for_classes = {}
        for clas in classes:
            try: metric_for_classes[clas] = values_metrics[clas][name_metric]
            except: metric_for_classes[clas] = None

        
        if os.path.exists(filename): ## If the file exists, load the data
            with open(filename, 'r') as f:
                previous_data = json.load(f)
                ## Differents cases

                if name_metric not in list(previous_data.keys()): ## New metric
                    skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"] = metric_for_classes
                    skeleton_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"] = metric_for_classes
                    skeleton_data[name_metric][name_model]["mean_value_metrics"] = metric_for_classes

                    previous_data[name_metric] = skeleton_data[name_metric]


                elif name_model not in list(previous_data[name_metric].keys()): ## New model to same metric
                    skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"] = metric_for_classes
                    skeleton_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"] = metric_for_classes
                    skeleton_data[name_metric][name_model]["mean_value_metrics"] = metric_for_classes

                    previous_data[name_metric][name_model] = skeleton_data[name_metric][name_model]


                else: ## Model and metric exists in the database
                    if name_dataset not in list(previous_data[name_metric][name_model]["dataset"].keys()): ## New dataset to same model and metric
                        skeleton_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"] = metric_for_classes
                        skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"] = metric_for_classes
                        previous_data[name_metric][name_model]["dataset"][name_dataset] = skeleton_data[name_metric][name_model]["dataset"][name_dataset]
                        previous_data[name_metric][name_model]["dataset"][name_dataset][name_gt] = skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]
                        ## Recalculate the mean value of the model
                        previous_data[name_metric][name_model]["mean_value_metrics"] = self.__recalculate_mean_values_model(previous_data, 
                                                                                                                    name_metric, 
                                                                                                                    name_model, 
                                                                                                                    classes
                                                                                                                    )
                    

                    elif name_gt not in list(previous_data[name_metric][name_model]["dataset"][name_dataset].keys()): ## New image to same dataset, model and metric
                        ## Recalculate the mean value of the dataset and the model
                        skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"] = metric_for_classes
                        previous_data[name_metric][name_model]["dataset"][name_dataset][name_gt] = skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]
                        ## Recalculate the mean value of the dataset
                        previous_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"] = self.__recalculate_mean_values_dataset(previous_data, 
                                                                                                                                        name_metric, 
                                                                                                                                        name_model, 
                                                                                                                                        name_dataset, 
                                                                                                                                        classes
                                                                                                                                        )
                        ## Recalculate the mean value of the model
                        previous_data[name_metric][name_model]["mean_value_metrics"] = self.__recalculate_mean_values_model(previous_data, 
                                                                                                                    name_metric, 
                                                                                                                    name_model, 
                                                                                                                    classes
                                                                                                                    )

                    else: ## Image exists in the database. Rewrite the values
                        ## Recalculate the mean value of the dataset and the model
                        skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["name_prediction"] = name_pred
                        skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"] = metric_for_classes
                        previous_data[name_metric][name_model]["dataset"][name_dataset][name_gt] = skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]
                        ## Recalculate the mean value of the dataset
                        previous_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"] = self.__recalculate_mean_values_dataset(previous_data, 
                                                                                                                                        name_metric, 
                                                                                                                                        name_model, 
                                                                                                                                        name_dataset, 
                                                                                                                                        classes
                                                                                                                                        )
                        ## Recalculate the mean value of the model
                        previous_data[name_metric][name_model]["mean_value_metrics"] = self.__recalculate_mean_values_model(previous_data, 
                                                                                                                    name_metric, 
                                                                                                                    name_model, 
                                                                                                                    classes
                                                                                                                    )
            with open(filename, 'w') as f:
                    json.dump(previous_data, f, indent=4)

        else: ## If the file does not exist, create a new one
            with open(filename, 'w') as f:
                ## New file same values metrics for all means
                skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"] = metric_for_classes
                skeleton_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"] = metric_for_classes
                skeleton_data[name_metric][name_model]["mean_value_metrics"] = metric_for_classes

                json.dump(skeleton_data, f, indent=4)


    @staticmethod
    def __get_skeleton_data(name_pred,name_gt, name_dataset, name_model, name_metric, dimensions):
        skeleton_string_data = {
            name_metric: {
                name_model: {
                    "dimensions": dimensions,
                    #"mean_value_metric": {}, # Change
                    "dataset": {
                        name_dataset: {
                            #"mean_value_metric": {}, # Change
                            name_gt: {
                                #"mean_value_metric": {},
                                "name_prediction": name_pred,
                                #"metrics": {} # change
                            }
                        }
                    }
                }
            }
        
        }

        return skeleton_string_data


    @staticmethod
    def __recalculate_mean_values_model(skeleton_data, name_metric, name_model, classes): 
        sums = {clas: 0 for clas in classes}
        counts = {clas: 0 for clas in classes}  

        ## Iter over all datasets
        for name_dataset in skeleton_data[name_metric][name_model]["dataset"]:
            ## Iter over all classes
            if "mean_value_metrics" in skeleton_data[name_metric][name_model]["dataset"][name_dataset]:
                for clas, metrics in skeleton_data[name_metric][name_model]["dataset"][name_dataset]["mean_value_metrics"].items():
                    if clas not in sums:
                        sums[clas] = 0
                        counts[clas] = 0
                    if metrics is not None:
                        sums[clas] += metrics
                        counts[clas] += 1

        ## Calculate the means
        return {clas: (total / counts[clas]) if counts[clas] > 0 else None for clas, total in sums.items()}


    @staticmethod
    def __recalculate_mean_values_dataset(skeleton_data, name_metric, name_model, name_dataset, classes): 
        sums = {clas: 0 for clas in classes}
        counts = {clas: 0 for clas in classes}  

        ## Iter over all datasets
        for name_gt in skeleton_data[name_metric][name_model]["dataset"][name_dataset]:
            ## Iter over all classes 
            if "metrics" in skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]:
                for clas, metrics in skeleton_data[name_metric][name_model]["dataset"][name_dataset][name_gt]["metrics"].items():
                    if clas not in sums:
                        sums[clas] = 0
                        counts[clas] = 0

                    if metrics is not None:
                        sums[clas] += metrics
                        counts[clas] += 1

        ## Calculate the means
        return {clas: (total / counts[clas]) if counts[clas] > 0 else None for clas, total in sums.items()}


if "__main__" == __name__:

    metrics = ["HD", "DICE", "JACCARD"]
    classes = ["liver", "kidney", "spleen"]
    battery_images = ["img0001", "img0002", "img0003"]
    pred_names = ["img0001_Pred", "img0002_Pred", "img0003_Pred"]
    dataset = "BTCV"
    models = ["unetr", "unet++", "vnet"]
    dimensions = 3
    values_metrics = {"liver": {
                        "HD": None,
                        "DICE": 1,
                        "JACCARD": 1
                    },
                    "kidney": {
                        "HD": 3,
                        "DICE": 3,
                        "JACCARD": 4
                    },
                    "spleen": {
                        "HD": 3,
                        "DICE": 2,
                        "JACCARD": 1
                    }
                
                }
    

    name_json = "metrics_test.json"

    SaveMetricsJson()(
            name_json, 
            pred_names[1], 
            battery_images[1], 
            dimensions, 
            dataset, 
            models[1], 
            metrics[0], 
            classes, 
            values_metrics
            )
