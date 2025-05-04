import torch
from torch.nn import functional as F
from typing import Dict, Tuple, List, Optional, Union, Callable
from detectron2.structures import ImageList, Instances
from dinoteacher.engine.build_dino import DinoVitFeatureExtractor
from detectron2.modeling.backbone import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
def build_dino_vit_backbone(cfg, _):
    return DinoVitFeatureExtractor_wrapper(cfg)


class DinoVitFeatureExtractor_wrapper(DinoVitFeatureExtractor):
    def __init__(self, cfg, output_layer='dino_out'):
        if 'dino' in cfg.MODEL.BACKBONE.NAME and cfg.SEMISUPNET.DINO_BBONE_LR_SCALE:
            freeze = False
        else:
            freeze = True
        super(DinoVitFeatureExtractor_wrapper, self).__init__(cfg, model_name=cfg.SEMISUPNET.DINO_BBONE_MODEL, normalize_feature=False, freeze=freeze, is_BGR=cfg.INPUT.FORMAT)
        self.output_layer = output_layer

    def forward(self, x):
        # The preprocessing is already done, but uses the BGR order by default,
        # while the DINO weights use RGB by default 
        x = x[:,[2,1,0],:,:]
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

        return {self.output_layer: x_grid_features}