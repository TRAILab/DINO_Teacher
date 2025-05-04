#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer
from dinoteacher import add_dinoteacher_config
from dinoteacher.engine.trainer import DINOTeacherTrainer
from dinoteacher.engine.gen_labels import test_and_gen

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

import dinoteacher.data.datasets.builtin
from dinoteacher.modeling.meta_arch.dino_vit import build_dino_vit_backbone
from dinoteacher.modeling.meta_arch.rcnn import DAobjTwoStagePseudoLabGeneralizedRCNN_shortcut
from dinoteacher.modeling.roi_heads.roi_heads import SingleScaleROIHeadsPseudoLab
from random import randint

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dinoteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "dinoteacher":
        Trainer = DINOTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "adapteacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.gen_labels:
        if cfg.SEMISUPNET.Trainer == "dinoteacher" or cfg.SEMISUPNET.Trainer == "adapteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = test_and_gen(cfg, ensem_ts_model.modelTeacher, parent_build_test_loader=Trainer.build_test_loader, parent_build_evaluator=Trainer.build_evaluator)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    elif args.eval_only:
        if cfg.SEMISUPNET.Trainer == "dinoteacher" or cfg.SEMISUPNET.Trainer == "adapteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--gen-labels", action="store_true", help="evaluate and store predicted instances")
    args = parser.parse_args()
    # args.gen_labels = True
    args.resume = True
    args.num_gpus = 1
    args.opts = ['SEMISUPNET.LABELER_TARGET_PSEUDOGT', 'output/dino_label/test_vitl/predictions/ACDC_val_night_dino_anno_vitl.pkl']
    args.config_file = './configs/vgg_city2bdd_test.yaml'
    print("Command Line Args:", args)
    url_parts = args.dist_url.rsplit(':',1)
    url_parts[1] = str(randint(0,1000) + int(url_parts[1]))
    args.dist_url = (':').join(url_parts)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )