import json
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from classMetrics import (
    RemovirtMetrics,
    MetricResult,
)

class ShowMetrics:
    models_3d = [
        'attention_unet',
        'medformer', 
        'resunet', 
        'swin_unetr', 
        'unet++', 
        'unetr', 
        'vnet', 
        'segformer'
    ]

    file_name = './metrics.json'

    all_classes = [ '__BKG__','Spleen','Right Kidney','Left Kideny','Gallbladder',
                'Esophagus','Liver', 'Stomach','Aorta','IVC','Portal and Splenic Veins',
                'Pancreas','Right adrenal gland','Left adrenal gland']

    all_metrics = MetricResult.metrics

    range_metrics = MetricResult.range_metrics


    def __init__(self):
        if os.path.exists('.'+self.file_name):
            with open('.'+self.file_name, 'r') as f:
                self.database = json.load(f)


    def graph_mean(self, metric_name):
        pass

    def heat_map(self, metric_name):
        pass

    def graph_class(self, metric_name, class_name, dataset_name="BTCV", output_dir="graph_classes"):
        data_metric = self.database[metric_name]
        models_names = list(data_metric.keys())  
        len_total_models = len(models_names)
        list_colors = cm.tab10(np.linspace(0, 1, len_total_models))

        min_value = 999999999999999999 # np.inf
        max_value = 0

        for i, model in enumerate(models_names):
            ## mean_value_model = data_metric[model]["mean_value_metrics"][class_name]
            ## mean_value_dataset = data_metric[model]["dataset"][dataset_name]["mean_value_metrics"][class_name]
            
            image_names = [image for image in data_metric[model]["dataset"][dataset_name] if "img" in image]

            data_values_images = {img: data_metric[model]["dataset"][dataset_name][img]["metrics"][class_name] for img in image_names} 
            
            image_names = list(data_values_images.keys())
            metric_values = list(data_values_images.values())

            try:
                min_value = min(metric_values) if min(metric_values) < min_value else min_value
                max_value = max(metric_values) if max(metric_values) > max_value else max_value
            except:
                print(f"Error in {model} model, {class_name} class and {dataset_name} dataset, all NaN values.")
                continue

            plt.plot(
                image_names, 
                metric_values,
                marker='o', 
                linestyle='-', 
                color=list_colors[i])

        ## Set metric range
        min_value = 0 if min_value == 999999999999999999 else min_value
        plt.ylim(min_value, max_value)
        ## Set legend
        plt.legend()
        custom_legend = [plt.Line2D([0], [0], color=list_colors[i], lw=2) for i in range(len_total_models)]
        plt.legend(custom_legend, models_names)
        ## Set labels
        plt.xlabel('Images Test')
        plt.ylabel('Metric Value')
        plt.title('Metric {} in range {} for the class {}'.format(metric_name, self.range_metrics[metric_name], class_name))
        plt.xticks(rotation=45)
        plt.tight_layout() 
        ## Save plot with high quality
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + f'/graph_{metric_name}_{class_name}.png', dpi=300)
        plt.close()


    def graph_means_models(self, metric_name, output_dir="graph_means"):
        data_metric = self.database[metric_name]
        models_names = list(data_metric.keys())  
        len_total_models = len(models_names)
        list_colors = cm.tab10(np.linspace(0, 1, len_total_models))

        min_value = 999999999999999999 # np.inf
        max_value = 0

        for i, model in enumerate(models_names):
            mean_value_model = data_metric[model]["mean_value_metrics"]
            image_names = list(mean_value_model.keys())
            metric_values = list(mean_value_model.values())
            try:
                filtered_values = [value for value in metric_values if value is not None]
                min_value = min(filtered_values) if min(filtered_values) < min_value else min_value
                max_value = max(filtered_values) if max(filtered_values) > max_value else max_value
            except:
                print(metric_values)
                print(f"Error in {model} model, all NaN values.")
                

            plt.plot(
                image_names, 
                metric_values,
                marker='o', 
                linestyle='-', 
                color=list_colors[i])

        ## Set metric range
        min_value = 0 if min_value == 999999999999999999 else min_value
        plt.ylim(min_value, max_value)
        ## Set legend
        plt.legend()
        custom_legend = [plt.Line2D([0], [0], color=list_colors[i], lw=2) for i in range(len_total_models)]
        plt.legend(custom_legend, models_names)
        ## Set labels
        plt.xlabel('Classes')
        plt.ylabel('Metric Value')
        
        plt.title('Metric {} in range {} means in the different models'.format(metric_name, self.range_metrics[metric_name]))

        plt.xticks(rotation=45)
        plt.tight_layout() 
        ## Save plot with high quality
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + f'/graph_{metric_name}_means.png', dpi=300)
        plt.close()


    def graph_histogram_mean(self, class_name, output_dir="graph_histogram"):
        metrics_names = list(self.database.keys())

        for metric_idx, metric in enumerate(metrics_names):
            data_metric = self.database[metric]
            models_names = list(data_metric.keys())  
            len_total_models = len(models_names)
            list_colors = cm.tab10(np.linspace(0, 1, len_total_models))
            
            bar_width = 0.2
            offset = np.linspace(-bar_width, bar_width, len(models_names))
            metric_position = metric_idx * len(models_names)
            for i, model in enumerate(models_names):
                mean_value_model = data_metric[model]["mean_value_metrics"][class_name]
                plt.bar(
                    metric_position + offset[i] + i * bar_width, 
                    mean_value_model,
                    bar_width,
                    color=list_colors[i],
                    label=model)

        plt.legend()
        custom_legend = [plt.Line2D([0], [0], color=list_colors[i], lw=2) for i in range(len_total_models)]
        plt.legend(custom_legend, models_names)

        plt.xlabel('Metrics Names')
        plt.ylabel('Metrics Value')
        
        plt.title('Histogram means all metrics with specific {} class'.format(class_name))

        #plt.xticks(rotation=45)
        plt.xticks(np.arange(len(metrics_names)) * len(models_names) + bar_width / 2, metrics_names, rotation=45)
        plt.tight_layout() 
        ## Save plot with high quality
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + f'/graph_{class_name}_means_all_metrics.png', dpi=300)
        plt.close()


    def graph_histogram_metric_mean(self, metric_name, class_name, output_dir="graph_histogram"):
        metrics_names = list(metric_name)

        data_metric = self.database[metric_name]
        models_names = list(data_metric.keys())  
        len_total_models = len(models_names)
        list_colors = cm.tab10(np.linspace(0, 1, len_total_models))
        bar_width = 2.0
        
        for i, model in enumerate(models_names):
            mean_value_model = data_metric[model]["mean_value_metrics"][class_name]
            plt.hist(
                mean_value_model,
                bins=10,
                color=list_colors[i],
                label=model,
                alpha=0.5
            )
            
        plt.legend()
        custom_legend = [plt.Line2D([0], [0], color=list_colors[i], lw=2) for i in range(len_total_models)]
        plt.legend(custom_legend, models_names)

        plt.xlabel('Metrics Names')
        plt.ylabel('Metrics Value')
        
        plt.title('Histogram means for {} metric with specific {} class'.format(metric_name, class_name))

        plt.xticks(np.arange(len(models_names)) * len(models_names) + bar_width / 2, models_names, rotation=45)
        plt.tight_layout() 
        ## Save plot with high quality
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + f'/graph_{metric_name}_{class_name}_means.png', dpi=300)
        plt.close()



if "__main__" == __name__:
    show_metrics = ShowMetrics()
    print("All metrics available: ", show_metrics.all_metrics)
    #show_metrics.graph_histogram_mean("Liver")
    
    for metric in show_metrics.all_metrics:
        print(f"Creating graphs for {metric} metric... ")
        show_metrics.graph_histogram_metric_mean(metric, "Liver")
    #    show_metrics.graph_class(metric, "Liver")
    #    show_metrics.graph_means_models(metric)

    #show_metrics.si()
    print("Finish!")