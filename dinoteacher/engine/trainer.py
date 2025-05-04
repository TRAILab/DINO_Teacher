"""
Based on the trainer.py file from the original Adaptive Teacher codebase.
See file at https://github.com/facebookresearch/adaptive_teacher/blob/main/adapteacher/engine/trainer.py
"""

import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.data import MetadataCatalog

from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from dinoteacher.data.dataset_mapper import DatasetMapperTwoCropSeparateKeepTf
from dinoteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from dinoteacher.engine.align_head import TeacherStudentAlignHead
from dinoteacher.engine.build_dino import DinoVitFeatureExtractor

from adapteacher.engine.probe import OpenMatchTrainerProbe
import copy
import pickle

# DINO Teacher model initialization and trainer. Based on ATeacherTrainer from Adaptive Teacher
class DINOTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):
        """
        Args:
            cfg (CfgNode):
            wandb_run (wandb.run): wandb run for logging
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # Adaptive Teacher specific domain invariance
        self.use_adversarial_invariance = cfg.SEMISUPNET.DIS_LOSS_WEIGHT > 0
        self.branch = 'supervised'
        if cfg.SEMISUPNET.USE_FEATURE_ALIGN:
            # If DINO alignment is used, add the model
            self.align_layer = cfg.SEMISUPNET.FEATURE_ALIGN_LAYER
            self.use_feature_align = True
            self.student_align_feat = {}
            student_align_dim = model.backbone._out_feature_channels[cfg.SEMISUPNET.FEATURE_ALIGN_LAYER]
            model.align_teacher = DinoVitFeatureExtractor(cfg, model_name=cfg.SEMISUPNET.ALIGN_MODEL, normalize_feature=cfg.SEMISUPNET.ALIGN_HEAD_NORMALIZE).eval()
            teacher_align_dim = [*model.align_teacher.modules()][-2].normalized_shape[0]
            model.align_student_head = TeacherStudentAlignHead(cfg, student_align_dim, teacher_align_dim, normalize_feature=model.align_teacher.normalize_feature)
            self._register_input_hook_feat_align(model, 'proposal_generator')

            model.align_teacher = model.align_teacher.to((torch.device(model.device)))
            model.align_student_head = model.align_student_head.to((torch.device(model.device)))   
            # self.align_easy_augs_only = cfg.SEMISUPNET.ALIGN_EASY_ONLY
            # self.align_target_iter = cfg.SEMISUPNET.FEATURE_ALIGN_TARGET_START
        else:
            self.use_feature_align = False


        if type(cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT) == str:
            file_in = cfg.SEMISUPNET.LABELER_TARGET_PSEUDOGT
            self.use_dino_PL = True
            with open(file_in, 'rb') as f_in:
                temp_dict = pickle.load(f_in)
            self.dino_pseudogt = {}
            for img in temp_dict:
                self.dino_pseudogt[img['image_id']] = img
        else:
            self.use_dino_PL = False

        if type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP) == str:
            assert type(cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER) == int
            self.PL_swap = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP
            self.PL_swap_iter = cfg.SEMISUPNET.LABELER_PSEUDOGT_SWAP_ITER
        else:
            self.PL_swap = None

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            if self.use_feature_align:
                model.align_teacher = model.module.align_teacher
                model.align_student_head = model.module.align_student_head
                model.forward_backbone = model.module.forward_backbone


        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # self.model.align_teacher.eval()

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

        # if wandb_run is not None:
        #     self.log_wandb = True
        #     self.wandb_run = wandb_run
        # else:
        #     self.log_wandb = False

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        print('\n\n\n\n',self.cfg.MODEL.WEIGHTS,'\n\n\n\n')
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparateKeepTf(cfg, is_train=True, keep_tf_data=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # Pseudo-Label Selection
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        elif proposal_type == "dino":
            valid_map = proposal_bbox_inst.gt_scores > thres
            new_proposal_inst = proposal_bbox_inst[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    # Training loop
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "Model in eval mode while training, set it to train mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # We instantiate and update the EMA model from initialization
        if self.iter % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(
                keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        # 3 stages: 
        #   1. source-only supervised training + source-only alignment with DINO
        #   2. source-only supervised training + source and target alignment with DINO
        #   3. supervised training with source gt and target pseudo labels + alignment

        record_dict = {}
        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q + unlabel_data_k
            self.branch = "supervised"
            record_dict, _, _, _ = self.model(
                all_label_data, branch="supervised")

            has_target_backbone_feats = 0                

        else:
            #  Remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            # Check if labeller pseudo-labels are used
            use_DT_labels = False
            if self.use_dino_PL:
                use_DT_labels = True
                if self.PL_swap == 'full':
                    if self.iter > self.PL_swap_iter:
                        use_DT_labels = False
                elif self.PL_swap == 'half':
                    if self.iter > self.PL_swap_iter and self.iter % 2:
                        use_DT_labels = False

            if use_DT_labels:
                # Use VFM labeller pseudo-labels
                instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] for x in unlabel_data_q]
                boxes = [(x['tf_data'].apply_box(y.pred_boxes),y.scores,y.pred_classes) for x,y in zip(unlabel_data_q,instances)]
                dino_pseudo_labels = []
                for i in range(len(instances)):
                    new_instances = Instances(unlabel_data_k[i]['image'].shape[-2:])
                    new_instances.gt_boxes = Boxes(boxes[i][0])
                    new_instances.gt_scores = boxes[i][1]
                    new_instances.gt_classes = boxes[i][2]
                    dino_pseudo_labels.append(new_instances)
                
                joint_proposal_dict = {}
                pseudo_proposals_dino, num_pseudo_bbox_roih = self.process_pseudo_label(
                    dino_pseudo_labels, cur_threshold, "dino", "thresholding")
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_dino

            else:
                # Generate the pseudo-labels using Mean Teacher EMA model
                self.branch = "unsup_data_weak"
                with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                #  Select pseudo-labels
                joint_proposal_dict = {}

                # Pseudo_labeling with RPN head (bbox location/objectness)
                # joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
                # #Process pseudo labels and thresholding
                # pseudo_proposals_rpn_unsup_k, num_pseudo_bbox_rpn, = self.process_pseudo_label(
                #     proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding")
                # joint_proposal_dict["proposals_pseudo_rpn"] = pseudo_proposals_rpn_unsup_k

                # Pseudo_labeling for ROI head (bbox location/objectness)
                pseudo_proposals_roih_unsup_k, num_pseudo_bbox_roih = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_roih_unsup_k

            # Add pseudo-labels and define datasets
            all_label_data = label_data_q + label_data_k
            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"])
            if use_DT_labels:
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"])
                all_unlabel_data = unlabel_data_q + unlabel_data_k
            else:
                all_unlabel_data = unlabel_data_q

            # Input both strongly and weakly augmented labeled data into student model
            self.branch = "supervised"
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised")
            record_dict.update(record_all_label_data)

            # Input unlabeled data into model (if VFM labels, use both weak and strong)
            self.branch = "supervised_target"
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target")
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = \
                    record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)

            # Target image backbone features available for all (2) or hard (1) images
            if use_DT_labels:
                has_target_backbone_feats = 2
            else:
                has_target_backbone_feats = 1

            if self.use_adversarial_invariance:
                # Input weakly labeled data (source) and weakly unlabeled data (target) to student model
                # give sign to the target data
                for i_index in range(len(unlabel_data_k)):
                    # unlabel_data_item = {}
                    for k, v in unlabel_data_k[i_index].items():
                        # label_data_k[i_index][k + "_unlabeled"] = v
                        label_data_k[i_index][k + "_unlabeled"] = v
                    # unlabel_data_k[i_index] = unlabel_data_item

                all_domain_data = label_data_k
                # all_domain_data = label_data_k + unlabel_data_k
                self.branch = "domain"
                record_all_domain_data, _, _, _ = self.model(
                    all_domain_data, branch="domain")
                record_dict.update(record_all_domain_data)
        
        # Feature aligment with teacher model
        if self.use_feature_align:
            # Source images
            if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                # use only weakly augmented images for the alignment target
                easy_feat = self.model.align_teacher(label_data_k)
                teacher_feat = easy_feat.repeat(2,1,1,1)
            else:
                teacher_feat = self.model.align_teacher(all_label_data)
            student_feat = self.model.align_student_head(
                self.student_align_feat['supervised'], teacher_feat.shape[2:])
            align_loss = self.model.align_student_head.align_loss(student_feat, teacher_feat)
            record_dict['loss_align'] = align_loss

            if self.iter >= self.cfg.SEMISUPNET.FEATURE_ALIGN_TARGET_START:
                # has_target_backbone_feats == 0: No target images parsed
                # has_target_backbone_feats == 1: Only strong target images to student model
                # has_target_backbone_feats == 2: All target images to student model

                if self.cfg.SEMISUPNET.ALIGN_EASY_ONLY:
                    # use only weakly augmented images for the alignment target
                    easy_feat_target = self.model.align_teacher(unlabel_data_k)
                    if has_target_backbone_feats == 0 or has_target_backbone_feats == 2:
                        teacher_feat_target = easy_feat_target.repeat(2,1,1,1)
                    else:
                        teacher_feat_target = easy_feat_target
                else:
                    teacher_feat_target = self.model.align_teacher(all_unlabel_data)

                if has_target_backbone_feats == 0:
                    all_unlabel_data = unlabel_data_q + unlabel_data_k
                    backbone_feat_target = self.model.forward_backbone(all_unlabel_data)[self.cfg.SEMISUPNET.FEATURE_ALIGN_LAYER] 
                    student_feat_target = self.model.align_student_head(
                        backbone_feat_target, teacher_feat_target.shape[2:])
                else:
                    student_feat_target = self.model.align_student_head(
                        self.student_align_feat['supervised_target'], teacher_feat_target.shape[2:])
                align_loss_target = self.model.align_student_head.align_loss(
                    student_feat_target, teacher_feat_target)
                record_dict['loss_align_target'] = align_loss_target


        # weight losses
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                    # pseudo bbox regression <- 0
                    loss_dict[key] = record_dict[key] * 0
                elif key[-6:] == "pseudo":  # unsupervised loss
                    loss_dict[key] = (
                        record_dict[key] *
                        self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    )
                elif (
                    key == "loss_D_img_s" or key == "loss_D_img_t"
                ):  # set weight for discriminator
                    # import pdb
                    # pdb.set_trace()
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                elif key == "loss_align":
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT
                elif key == "loss_align_target":    
                    loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.FEATURE_ALIGN_LOSS_WEIGHT_TARGET
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()


    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        # Uncomment the following line to enable evaluation on the student model
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
        #            test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _get_detector_input_hook(self, module, input, output):
        self.student_align_feat[self.branch] = input[1][self.cfg.MODEL.RPN.IN_FEATURES[0]]

    def _register_input_hook_feat_align(self, model, target_layer):
        for (name, module) in model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._get_detector_input_hook)
        return True
