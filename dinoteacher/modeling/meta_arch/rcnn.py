from typing import Tuple, Optional
import torch.nn as nn
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from adapteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut(DAobjTwoStagePseudoLabGeneralizedRCNN):
    def __init__(
        self,
        cfg
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut, self).__init__(cfg)
        

    # @classmethod
    # def from_config(cls, cfg):
    #     backbone = build_backbone(cfg)
    #     return {
    #         "backbone": backbone,
    #         "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
    #         "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
    #         "input_format": cfg.INPUT.FORMAT,
    #         "vis_period": cfg.VIS_PERIOD,
    #         "pixel_mean": cfg.MODEL.PIXEL_MEAN,
    #         "pixel_std": cfg.MODEL.PIXEL_STD,
    #         "dis_type": cfg.SEMISUPNET.DIS_TYPE,
    #     }

    def forward_backbone(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        return features