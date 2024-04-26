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

    def graph_class(self, metric_name, class_name, dataset_name="BTCV", ):
        data_metric = self.database[metric_name]
        models_names = list(data_metric.keys())  
        len_total_models = len(models_names)
        list_colors = cm.tab10(np.linspace(0, 1, len_total_models))

        for i, model in enumerate(models_names):
            mean_value_model = data_metric[model]["mean_value_metrics"][class_name]
            mean_value_dataset = data_metric[model]["dataset"][dataset_name]["mean_value_metrics"][class_name]
            
            image_names = [image for image in data_metric[model]["dataset"][dataset_name] if "img" in image]

            data_values_images = {img: data_metric[model]["dataset"][dataset_name][img]["metrics"][class_name] for img in image_names} 
            
            image_names = list(data_values_images.keys())
            metric_values = list(data_values_images.values())

            plt.plot(
                image_names, 
                metric_values,
                marker='o', 
                linestyle='-', 
                color=list_colors[i])

        ## Set metric range
        plt.ylim(self.range_metrics[metric_name])
        ## Set legend
        plt.legend()
        custom_legend = [plt.Line2D([0], [0], color=list_colors[i], lw=2) for i in range(len_total_models)]
        plt.legend(custom_legend, models_names)
        ## Set labels
        plt.xlabel('Images Test')
        plt.ylabel('Metric Value')
        plt.title('Metric {} for the class {} in the different models'.format(metric_name, class_name))
        plt.xticks(rotation=45)
        plt.tight_layout() 
        ## Save plot with high quality
        plt.savefig(f'graph_{metric_name}_{class_name}.png', dpi=300)

        plt.close()

    def graph_means_models(self, metric_name):
        data_metric = self.database[metric_name]
        models_names = list(data_metric.keys())  
        len_total_models = len(models_names)
        list_colors = cm.tab10(np.linspace(0, 1, len_total_models))

        for i, model in enumerate(models_names):
            mean_value_model = data_metric[model]["mean_value_metrics"]
            image_names = list(mean_value_model.keys())
            metric_values = list(mean_value_model.values())

            plt.plot(
                image_names, 
                metric_values,
                marker='o', 
                linestyle='-', 
                color=list_colors[i])

        ## Set metric range
        plt.ylim(self.range_metrics[metric_name])

        ## Set legend
        plt.legend()
        custom_legend = [plt.Line2D([0], [0], color=list_colors[i], lw=2) for i in range(len_total_models)]
        plt.legend(custom_legend, models_names)
        ## Set labels
        plt.xlabel('Classes')
        plt.ylabel('Metric Value')
        plt.title('Metric {} means in the different models'.format(metric_name))
        plt.xticks(rotation=45)
        plt.tight_layout() 
        ## Save plot with high quality
        plt.savefig(f'graph_{metric_name}_means.png', dpi=300)

        plt.close()


    def si(self):
        datos = [1, 2, 3, None, 5, 6, None, 8, 9, 10]
        datos = [None, None]

        # Filtrar los valores None
        datos_filtrados = [x for x in datos if x is not None]

        # Crear el histograma con los datos filtrados
        plt.hist(datos_filtrados, bins=10, color='skyblue')

        # Etiquetar los ejes y título
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Datos')

        plt.savefig('test.png', dpi=300)

        plt.close()

if "__main__" == __name__:
    show_metrics = ShowMetrics()
    print("All metrics available: ", show_metrics.all_metrics)
    for metric in show_metrics.all_metrics:
        show_metrics.graph_class(metric, "Liver")
        show_metrics.graph_means_models(metric)

    #show_metrics.si()
    print("Finish!")