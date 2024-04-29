import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
    """Function to get the model based on the arguments.
    Actually the models available, only 3 dimensions, are:
        - UNETR
        - UNet
        - VNet
        - AttentionUNet
        - ResUNet
        - MedFormer
        - SegFormer
        - UNetPlusPlus
        - SwinUNETR
        - SAM3D
        - nnFormer: don't work properly TODO
        - VTUNet: don't work properly TODO
        - FCN_Net: don't work properly TODO
    TODO: Implement the other models, for example:
        - DeepLabV3
        - PSPNet


    Args:
        args (argparse.Namespace): Arguments from the command line.
        pretrain (bool, optional): Set to true if you use a pretrained model. Defaults to False.

    Raises:
        ValueError: No pretrain model available
        ValueError: Invalid dimension, should be '2d' or '3d'

    Returns:
        Model: The model object.
    """    
    
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
        
        elif args.model == "segformer":
            from .dim3 import SegFormer3D
            return SegFormer3D(
                in_channels=args.in_chan,
                sr_ratios=args.sr_ratios,
                embed_dims=args.embed_dims,
                patch_kernel_size=args.patch_kernel_size,
                patch_stride=args.patch_stride,
                patch_padding=args.patch_padding,
                mlp_ratios=args.mlp_ratios,
                num_heads=args.num_heads,
                depths=args.depths,
                decoder_head_embedding_dim=args.decoder_head_embedding_dim,
                num_classes=args.classes,
                decoder_dropout=args.decoder_dropout,
            )

        elif args.model == 'sam':
            from .dim3 import Sam3D
            return Sam3D(
                num_classes=args.classes, 
                ckpt=None, 
                image_size=args.crop_size, 
                vit_name=args.vit_name,
                num_modalities=args.in_chan, 
                do_ds=args.do_ds
                )

        elif args.model == 'dints':
            from .dim3 import DiNTS, TopologySearch, TopologyInstance
            # dints_space = TopologySearch(
            #     channel_mul=0.5,
            #     num_blocks=12,
            #     num_depths=4,
            #     use_downsample=True,
            #     device=args.device,
            # )
            dints_space = TopologyInstance(
                channel_mul=args.channel_mul,
                num_blocks=args.num_blocks,
                num_depths=args.num_depths,
                use_downsample=args.use_downsample,
                device=args.aug_device,
            )
            return DiNTS(
                    dints_space=dints_space,
                    in_channels=args.in_chan,
                    num_classes=args.classes,
                    act_name=args.act_name,
                    norm_name=("INSTANCE", {"affine": True}),
                    spatial_dims=args.spatial_dims,
                    use_downsample=args.use_downsample,
                    node_a=None,
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

