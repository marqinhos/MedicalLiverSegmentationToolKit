import torch
import json
from monai.networks.nets import UNETR
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
import os
import yaml
import argparse
from xmlrpc.client import boolean

from model.utils import get_model



def get_parser(model):
    """Function to get the parser with the arguments.

    Raises:
        ValueError: The specified configuration doesn't exist

    Returns:
        argparse.Namespace: Arguments from the command line.
    """    
    parser = argparse.ArgumentParser(description="Framework to train, test and predict with different medical models")

    parser.add_argument("--max_epochs", default=900, type=int, help="Max number of epochs for training")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training")
    parser.add_argument("--cache_rate", default=1.0, type=float, help="Cache rate for training")
    parser.add_argument("--pin_memory", default=False, type=bool, help="Pin memory for training")

    parser.add_argument("--percentage_train", default=0.8, type=float, help="Percentage of training data")
    
    parser.add_argument("--spatial_dims", default=3, type=int, help="Numero de dimension espaciais (2D ou 3D)")
    parser.add_argument("--in_channels", default=1, type=int, help="Input image channels (i.e. 3 for color images, 1 for gray images)")
    parser.add_argument("--out_channels", default=14, type=int, help="Number of classes")
    parser.add_argument("--data_dir", default='../Datasets/BTCV_/', type=str, help="Training data directory")
    parser.add_argument("--mode", default='Predict', type=str, help="Work mode (Train, Test, Predict)")
    parser.add_argument("--trainmode", default='init', type=str, help="Continue training from checkpoint (cont) or start from scratch (init)")
    parser.add_argument("--roi_size", default=(96, 96, 96), type=tuple, help="Slide window size for inference")
    parser.add_argument("--inference_batch_size", default=1, type=int, help="Batch size for inference")
    parser.add_argument('--folders_img_lbl', type=bool, default=True, help="If images and labels are in different folders")
    
    parser.add_argument("--show", default=False, type=boolean, help="Visualizar resultados on-line")

    parser.add_argument('--model', type=str, default=model, help="Network model name. Available models: unet, unetr, swin_unet, unet++, attention_unet, resunet, medformer, vnet, segformer")
    parser.add_argument('--dimension', type=str, default='3d', help="Dimension of the model (2d or 3d)")
    parser.add_argument('--dataset', type=str, default='btcv', help="Name of the dataset")
    parser.add_argument('--run_version', type=int, default=2, help="Version of the checkpoint for testing or predicting")
    parser.add_argument('--path_prediction', type=str, default="./results/", help="Path to save the predictions")


    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    
    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)
    

    return args




def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return round(num, 3), '%s' % (['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def calculate_params(model):
    return sum(p.numel() for p in model.parameters())


def calculate_flops(model, input_tensor):
    flop = FlopCountAnalysis(model, input_tensor)
    return flop.total()


def estimate_memory_inference(
    model, sample_input, batch_size=1, use_amp=False, device=0
):
    """Predict the maximum memory usage of the model.
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a 

    model_input = sample_input  # .unsqueeze(0).repeat(batch_size, 1)
    output = model(model_input.to(device)).sum()
    total_memory = model_memory/ (1024 ** 2)
    return round(total_memory, 3), 'MB'


def write_data_json(data, net, num_param, num_flops, memory, feature_size, num_heads):
    data[net] = {}
    data[net]["num_param"] = (num_param[0], num_param[1])
    data[net]["num_flops"] = (num_flops[0], num_flops[1])
    data[net]["memory_use"] = (memory[0], memory[1])
    data[net]["feature_size"] = feature_size
    data[net]["num_heads"] = num_heads

    return data


def main(name_model, data):
    args = get_parser(name_model)
    model = get_model(args)

    model.eval()

    batch_size = args.batch_size
    channels = args.in_chan
    depth, height, width = args.training_size
    input_tensor = torch.randn(batch_size, channels, depth, height, width)
    
    num_params = calculate_params(model)
    num_flops = calculate_flops(model, input_tensor)
    memory = estimate_memory_inference(model, input_tensor)
    feature_size = args.base_chan
    try:
        num_heads = args.num_heads
    except:
        num_heads = [1]
    
    data = write_data_json(
        data, 
        name_model, 
        human_readable(num_params), 
        human_readable(num_flops), 
        memory, 
        feature_size, 
        num_heads
        )
    #data = [human_readable(num_params), human_readable(num_flops), memory, feature_size, num_heads]

    return data



if __name__ == '__main__':
    
    name_models = [
        'attention_unet', 
        'medformer',
        'resunet',
        'swin_unetr',
        'unet++',
        'unetr',
        'vnet',
        'segformer',
        'unet',
        'dints'
        ]
    
    data = {}
    for name in name_models:
        data = main(name, data)
    #data = main(name_models[0], data)
    print(data)
    # Save file 
    with open('networks.json', 'w') as f:
        json.dump(data, f, indent=5)



    #flop = FlopCountAnalysis(model, input_tensor)
    #table = flop_count_table(flop, max_depth=1)


    #print("Number of params: %d"%num_param)
    #print("Number de FLOPs: %d"%human_readable(num_flops))
    #print(human_readable(num_flops))

    #print(flop_count_table(flop, max_depth=1))


    

