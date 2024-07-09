# Quickstart

```{warning}
This project is under development
```

## Installation

```{admonition} Software installation
Download Medical Liver Segmentation ToolKit
```

``` {code-block}
(.venv) $ git clone https://github.com/Removirt/LiverSegmentation.git

```

``` {code-block}
(.venv) $ cd MedicalLiverSegmentationToolkit/

```

``` {code-block}
(.venv) $ pip install -r requirements.txt

```


## Usage
```{important}
To introduce new network architectures, follow these steps:

1. Enter the architecture in `model/dim3/{your_architecture}.py`.
2. Add the network import in `model/utils.py`.
3. Add the network training configuration in `config/{database}/{your_architecture}_3d.yaml`.
```



##### Training Network
``` {code-block}
(.venv) $ python3 train.py --model {network_name} --max_epochs {num_max_epochs}
```

```{note}
See the [Train Module](../modules/train) documentation for more info on parameters.
```

```{tip}
To train more than 1 network, use `train_sequential.py`, [more info](../modules/train_sequential).
```

##### Test Network
``` {code-block}
(.venv) $ python3 train.py --model {network_name} --version {training_version}
```

##### Predict Network
``` {code-block}
(.venv) $ python3 train.py --model {network_name} --version {training_version}
```
```{tip}
To predict more than 1 network, use `predict_sequential.py`, [more info](../modules/predict_sequential).
```


## Evaluation trained models

#### Performance measures
```{note}
The performance measures compute are:

1. Dice Similarity Coefficient (DSC).
2. Normalize Surface Distance (NSD).
3. Mean Average Surface Distance (MASD).
4. Hausdorff Distance (HD).
5. Relative Volume Difference (RVD).

```

``` {code-block}
(.venv) $ python3 metrics_sequential.py 
```

```{note}
Generate a JSON file with networks put in ([more info](../modules/metrics_sequential)):
```{code-block} python
:lineno-start: 20
:emphasize-lines: 10

models_3d = [
        'attention_unet',
        'medformer', 
        'resunet', 
        'swin_unetr', 
        'unet++', 
        'unetr', 
        'vnet', 
        'segformer',
        '{name_architecture}',
    ]
```


#### Complexity measures

``` {code-block}
(.venv) $ python3 calculate_features_networks.py 
```

```{note}
The profiler calculate:

1. Number of params.
2. Floating point operations per second (Flops).
3. Memory usage in a inference.
4. Layers size.


[more info](../modules/calculate_features_networks).
```



