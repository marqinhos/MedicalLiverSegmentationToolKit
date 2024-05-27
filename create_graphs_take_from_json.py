import json

def take_value4metric(measure="RVD"):
    # Cargar el archivo JSON
    with open('metrics.json', 'r') as file:
        data = json.load(file)

    # Lista para guardar los valores de "Liver" por red
    liver_values_dict = {
        'resunet': None,
        'unet++': None,
        'attention_unet': None,
        'unetr': None,
        'swin_unetr': None,
        'medformer': None,
        'segformer': None
    }

    # Recorrer el JSON para extraer los valores de "Liver"
    for network, network_data in data[measure].items():
        liver_value = None
        if "mean_value_metrics" in network_data:
            liver_value = network_data["mean_value_metrics"].get("Liver")
        for dataset, dataset_data in network_data.get("dataset", {}).items():
            for img, img_data in dataset_data.items():
                if "mean_value_metrics" in img_data:
                    liver_value = img_data["mean_value_metrics"].get("Liver")
        # Mapear el nombre de la red en el JSON a uno de los nombres en liver_values_dict
        if network.lower().replace(' ', '_') in liver_values_dict:
            liver_values_dict[network] = round(liver_value, 3)

    # Extraer los valores en el orden especificado
    ordered_liver_values = [liver_values_dict[key] for key in liver_values_dict]

    print(ordered_liver_values)

    return ordered_liver_values


def take_param():
    with open('networks.json', 'r') as file:
        data = json.load(file)

    net_values_dict = {
        'unet': None,
        'vnet': None,
        'resunet': None,
        'unet++': None,
        'attention_unet': None,
        'unetr': None,
        'swin_unetr': None,
        'medformer': None,
        'segformer': None
    }

    for net_name in list(net_values_dict.keys()):
        net_values_dict[net_name] = data[net_name]["num_param"][0]

    print(net_values_dict)
    print(net_values_dict.values())
    print(net_values_dict.keys())

take_param()