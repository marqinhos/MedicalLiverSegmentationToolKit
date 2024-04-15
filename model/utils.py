import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
    

    if args.dimension == '3d':
        
        if args.model == 'unetr':
            from .dim3 import UNETR
            return UNETR(
                args.in_chan, 
                args.classes, 
                args.training_size, 
                feature_size=16, 
                hidden_size=768, 
                mlp_dim=3072, 
                num_heads=12, 
                pos_embed='perceptron', 
                norm_name='instance', 
                res_block=True
                )

        elif args.model == 'resunet':
            from .dim3 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )

    
        elif args.model == 'vnet':
            from .dim3 import VNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return VNet(
                args.in_chan, 
                args.classes, 
                scale=args.downsample_scale, 
                baseChans=args.base_chan
                )

        elif args.model == 'attention_unet':
            from .dim3 import AttentionUNet
            return AttentionUNet(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )

        elif args.model == 'unet':
            from .dim3 import UNet
            return UNet(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )
        
        elif args.model == 'medformer':
            from .dim3 import MedFormer
            return MedFormer(
                args.in_chan, 
                args.classes, 
                args.base_chan, 
                map_size=args.map_size, 
                conv_block=args.conv_block, 
                conv_num=args.conv_num, 
                trans_num=args.trans_num, 
                num_heads=args.num_heads, 
                fusion_depth=args.fusion_depth, 
                fusion_dim=args.fusion_dim, 
                fusion_heads=args.fusion_heads, 
                expansion=args.expansion, 
                attn_drop=args.attn_drop, 
                proj_drop=args.proj_drop, 
                proj_type=args.proj_type, 
                norm=args.norm, 
                act=args.act, 
                kernel_size=args.kernel_size, 
                scale=args.down_scale, 
                aux_loss=args.aux_loss
                )
    
        elif args.model == 'unet++':
            from .dim3 import UNetPlusPlus
            return UNetPlusPlus(
                args.in_chan, 
                args.base_chan, 
                num_classes=args.classes, 
                scale=args.down_scale, 
                norm=args.norm, 
                kernel_size=args.kernel_size, 
                block=args.block
                )
        
        elif args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            return SwinUNETR(
                args.window_size, 
                args.in_chan, 
                args.classes, 
                feature_size=args.base_chan
                )
            
        
        elif args.model == 'nnformer':
            from .dim3 import nnFormer
            return nnFormer(
                args.window_size, 
                input_channels=args.in_chan, 
                num_classes=args.classes, 
                deep_supervision=args.aux_loss
                )
            
        
        elif args.model == 'vtunet':
            from .dim3 import VTUNet
            return VTUNet(
                args, 
                args.classes
                )

        elif args.model == 'fcn_net':
            from .dim3 import FCN_Net
            return FCN_Net(
                args.in_chan, 
                args.classes, 
                )

    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')
