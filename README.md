# Framework that contains all most famous SOTAs Networks for abdominal organs segmentation
Networks available now [April 15, 2024] are:
 - UNET
 - UNETR
 - SwinUNET
 - UNET++
 - Medformer
 - Resunet
 - Attention UNET
 - VNet





# Known error with networks:
```bash
nnformer
return forward_call(*args, **kwargs)
  File "/home/marcos/Marcos_PhD/LiverSegmentation/.env_liver_segmentation/lib/python3.10/site-packages/monai/losses/dice.py", line 784, in forward
    if len(input.shape) != len(target.shape):
AttributeError: 'tuple' object has no attribute 'shape'

    if len(input.shape) != len(target.shape):
AttributeError: 'tuple' object has no attribute 'shape'
```

```bash
vtunet
    return forward_call(*args, **kwargs)
  File "/home/marcos/Marcos_PhD/LiverSegmentation/model/dim3/vtunet_utils.py", line 921, in forward
    x = x.view(B, 32 // self.D_ratio, H, W, C)
RuntimeError: shape '[1, 32, 6, 6, 768]' is invalid for input of size 663552
```