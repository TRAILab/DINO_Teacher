"""
Build DINO ViT backbone
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14, dinov2_vitb14_reg, dinov2_vitl14_reg
from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec
from typing import Dict

class dino_preprocessing():
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self, pixel_mean, pixel_std):
        normalize = T.Normalize(mean=pixel_mean, std=pixel_std)        
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)

class DinoVitFeatureExtractor(Backbone):
    """
    DINO V2 Vision Transformer Feature Extractor.
    """
    def __init__(self, cfg, model_name='dinov2_vits14', normalize_feature=True, freeze=True, is_BGR=True):
        super(DinoVitFeatureExtractor, self).__init__()
        # Pretrained DINO weights use ImageNet normalization for RGB images
        pixel_mean = [123.675, 116.280, 103.530]
        pixel_std = [58.395, 57.120, 57.375]
        self.preprocessing = dino_preprocessing(pixel_mean, pixel_std)

        self.is_BGR = is_BGR
        self.normalize_feature = normalize_feature
        if "v2" not in model_name:
            # 'dino_vitb16'
            self.model_name = model_name
            local_dir = "dinoteacher/engine/dinov1/hub/facebookresearch_dino_main"
            self.encoder = torch.hub.load(local_dir, source='local', model=model_name, path=model_name)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False                     
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = int(model_name.rsplit('vit',1)[-1][1:])
            assert (cfg.INPUT.DINO_PATCH_SIZE == self.patch_size), f'Config patch size is {cfg.INPUT.DINO_PATCH_SIZE} while loaded model has a patch size of {self.patch_size}'
        else:
            dino_v2_models = {
                "dinov2_vits14": (14, 384, dinov2_vits14), # patch_size, output dims, function name to create model
                "dinov2_vitb14": (14, 768, dinov2_vitb14),
                "dinov2_vitl14": (14, 1024, dinov2_vitl14),
                "dinov2_vitg14": (14, 1536, dinov2_vitg14),
                "dinov2_vitb14_reg4": (14, 768, dinov2_vitb14_reg),
                "dinov2_vitl14_reg4": (14, 1024, dinov2_vitl14_reg),
                # "dinov2_vitb16": (16, 768, dinov2_vitb14),
            }
            # model name to model weights
            name_to_weights = {"dinov2_vits14": "dinov2_vits14_pretrain.pth",
                            "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
                            "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
                            "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
                            "dinov2_vitb14_reg4": "dinov2_vitb14_reg4_pretrain.pth",
                            "dinov2_vitl14_reg4": "dinov2_vitl14_reg4_pretrain.pth",
                            # "dinov2_vitb16": "dinov2_vitb14_pretrain_14to16.pth",
            }
            # load model on cpu
            self.model_name = model_name
            assert (
                self.model_name in dino_v2_models.keys()
            ), f"class DinoV2VitFeatureExtractor(nn.Module): is only available for {dino_v2_models.keys()}"
            path_to_pretrained_weights = "weights/" + model_name + "_pretrain.pth"
            assert (
                os.path.exists(path_to_pretrained_weights)
            ), f"DINO v2 pretrained model path {path_to_pretrained_weights} does not exist!"
            print(f"Model Path: {path_to_pretrained_weights}")
            
            patch_size, embed_dim, model_func_name = dino_v2_models[self.model_name]
            # load model
            if patch_size == 16 and "v2" in model_name:
                img_size = 592
            else:
                img_size = 518  
            self.encoder = model_func_name(pretrained=False, patch_size=patch_size, img_size=img_size)
            self.encoder.load_state_dict(torch.load(path_to_pretrained_weights),strict=False)
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            # ensure 
            assert self.encoder.embed_dim == embed_dim
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = patch_size
            self._out_features = ['dino_out']
            self._out_feature_channels = {'dino_out':self.encoder.blocks[-1].norm2.bias.shape[0]}
            self._out_feature_strides = {'dino_out':self.patch_size}


    def forward(self, x):
        # the data loading defaults to BGR
        if self.is_BGR:
            x = torch.stack([img['image'] for img in x], dim=0)[:,[2,1,0],:,:].float()
        else:
            x = torch.stack([img['image'] for img in x], dim=0).float()
        x = self.preprocessing(x).to(device=next(self.encoder.parameters()).device)
        batch_size, _, height, width = x.size()
        # check image dims divisible by patch_size
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        x = self.encoder.get_intermediate_layers(x)[0] # batch_size, num_patches, self.embed_dim
        if "v2" not in self.model_name:
            x = x[:,1:,:] # remove class token

        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=2)

        x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        return x_grid_features
    
    @property
    def size_divisibility(self) -> int:
        return self.patch_size

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {'size_divisibility': self.size_divisibility, 'square_size': 0}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
