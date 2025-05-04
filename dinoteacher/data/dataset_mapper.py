# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from adapteacher.data.detection_utils import build_strong_augmentation

from fvcore.transforms.transform import (
    Transform,
)
import torch.nn.functional as F
from math import floor
from fvcore.transforms.transform import CropTransform, PadTransform, TransformList
from detectron2.data.transforms.augmentation import Augmentation
from cityscapesscripts.helpers.labels import id2label

# class DatasetMapperTwoCropSeparate(DatasetMapper):
#     """
#     This customized mapper produces two augmented images from a single image
#     instance. This mapper makes sure that the two augmented images have the same
#     cropping and thus the same size.

#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by the model.

#     This is the default callable to be used to map your dataset dict into training data.
#     You may need to follow it to implement your own one for customized logic,
#     such as a different way to read or transform images.
#     See :doc:`/tutorials/data_loading` for details.

#     The callable currently does the following:

#     1. Read the image from "file_name"
#     2. Applies cropping/geometric transforms to the image and annotations
#     3. Prepare data and annotations to Tensor and :class:`Instances`
#     """

#     def __init__(self, cfg, is_train=True):
#         self.augmentation = utils.build_augmentation(cfg, is_train)
#         # include crop into self.augmentation
#         if cfg.INPUT.CROP.ENABLED and is_train:
#             self.augmentation.insert(
#                 0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
#             )
#             logging.getLogger(__name__).info(
#                 "Cropping used in training: " + str(self.augmentation[0])
#             )
#             self.compute_tight_boxes = True
#         else:
#             self.compute_tight_boxes = False
#         self.strong_augmentation = build_strong_augmentation(cfg, is_train)

#         # fmt: off
#         self.img_format = cfg.INPUT.FORMAT
#         self.mask_on = cfg.MODEL.MASK_ON
#         self.mask_format = cfg.INPUT.MASK_FORMAT
#         self.keypoint_on = cfg.MODEL.KEYPOINT_ON
#         self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
#         # fmt: on
#         if self.keypoint_on and is_train:
#             self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
#                 cfg.DATASETS.TRAIN
#             )
#         else:
#             self.keypoint_hflip_indices = None

#         if self.load_proposals:
#             self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
#             self.proposal_topk = (
#                 cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
#                 if is_train
#                 else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
#             )
#         self.is_train = is_train

#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         # utils.check_image_size(dataset_dict, image)

#         if "sem_seg_file_name" in dataset_dict:
#             sem_seg_gt = utils.read_image(
#                 dataset_dict.pop("sem_seg_file_name"), "L"
#             ).squeeze(2)
#         else:
#             sem_seg_gt = None

#         aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
#         transforms = aug_input.apply_augmentations(self.augmentation)
#         image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
#         image_shape = image_weak_aug.shape[:2]  # h, w

#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

#         if self.load_proposals:
#             utils.transform_proposals(
#                 dataset_dict,
#                 image_shape,
#                 transforms,
#                 proposal_topk=self.proposal_topk,
#                 min_box_size=self.proposal_min_box_size,
#             )

#         if not self.is_train:
#             dataset_dict.pop("annotations", None)
#             dataset_dict.pop("sem_seg_file_name", None)
#             return dataset_dict

#         if "annotations" in dataset_dict:
#             for anno in dataset_dict["annotations"]:
#                 if not self.mask_on:
#                     anno.pop("segmentation", None)
#                 if not self.keypoint_on:
#                     anno.pop("keypoints", None)

#             annos = [
#                 utils.transform_instance_annotations(
#                     obj,
#                     transforms,
#                     image_shape,
#                     keypoint_hflip_indices=self.keypoint_hflip_indices,
#                 )
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=self.mask_format
#             )

#             if self.compute_tight_boxes and instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

#             bboxes_d2_format = utils.filter_empty_instances(instances)
#             dataset_dict["instances"] = bboxes_d2_format

#         # apply strong augmentation
#         # We use torchvision augmentation, which is not compatiable with
#         # detectron2, which use numpy format for images. Thus, we need to
#         # convert to PIL format first.
#         image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
#         image_strong_aug = np.array(self.strong_augmentation(image_pil))
#         dataset_dict["image"] = torch.as_tensor(
#             np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
#         )

#         dataset_dict_key = copy.deepcopy(dataset_dict)
#         dataset_dict_key["image"] = torch.as_tensor(
#             np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
#         )
#         assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
#         assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
#         return (dataset_dict, dataset_dict_key)


class DatasetMapperTwoCropSeparateKeepTf(DatasetMapper):
    """
    Modification of DatasetMapperTwoCropSeparate from Adaptive Teacher to
    keep the transform data, which is applied to the labeller proposals
    """

    def __init__(self, cfg, is_train=True, keep_tf_data=False):
        # Crop to patch size when using a ViT for backbone or alignment

        if cfg.SEMISUPNET.USE_FEATURE_ALIGN or 'vit' in cfg.MODEL.BACKBONE.NAME:
            crop_to_patch_size = True
            self.augmentation = augs_with_transformer_patch(cfg, is_train)
        else:
            crop_to_patch_size = False
            self.augmentation = utils.build_augmentation(cfg, is_train)
        # include crop into self.augmentation
        if cfg.INPUT.CROP.ENABLED and is_train:
            if crop_to_patch_size:
                self.augmentation.insert(
                    0, RandomCropAndPad(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
                )
            else:
                self.augmentation.insert(
                    0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
                )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )
            self.compute_tight_boxes = True
        else:
            self.compute_tight_boxes = False
        self.strong_augmentation = build_strong_augmentation(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self.keep_tf_data = keep_tf_data

        self.mask_format = 'polygon'
        self.mask_on = cfg.MODEL.MASK_ON

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]  # h, w

        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.copy()
            labelIds = np.unique(sem_seg_gt)
            # if self.seg_instances_only:
            #     class2id = {'person':0,'rider':1,'car':2,'truck':3,'bus':4,'train':5,'motorcycle':6,'bicycle':7}
            #     for labelId in labelIds:
            #         class_ = id2label[labelId].name
            #         trainId = class2id.get(class_, 255)
            #         mask = sem_seg_gt == labelId
            #         sem_seg_gt[mask] = trainId
            # else:
            for labelId in labelIds:
                trainId = id2label[labelId].trainId
                mask = sem_seg_gt == labelId
                sem_seg_gt[mask] = trainId
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            for i in range(len(annos)):
                if 'segmentation' in annos[i].keys() and self.mask_format == "polygon":
                    if type(annos[i]['segmentation']) != list:
                        annos[i]['segmentation'] = [annos[i]['segmentation']]
                    # for lv1 in range(len(annos)):
                    #     # annos[lv1]['segmentation'] = [annos[lv1]['segmentation'][0].reshape(-1,2)]
                    #     annos[lv1]['segmentation'] = [x.tolist() for x in annos[lv1]['segmentation']]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )

            if self.compute_tight_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            bboxes_d2_format = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = bboxes_d2_format

        if self.keep_tf_data:
            dataset_dict['tf_data'] = transforms

        # apply strong augmentation
        # We use torchvision augmentation, which is not compatiable with
        # detectron2, which use numpy format for images. Thus, we need to
        # convert to PIL format first.
        image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
        img_strong = self.strong_augmentation(image_pil)
        image_strong_aug = np.array(img_strong)
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        )

        dataset_dict_key = copy.deepcopy(dataset_dict)
        dataset_dict_key["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
        )
        assert dataset_dict["image"].size(1) == dataset_dict_key["image"].size(1)
        assert dataset_dict["image"].size(2) == dataset_dict_key["image"].size(2)
        return (dataset_dict, dataset_dict_key)

# class DatasetMapper_test(DatasetMapper):
#     """
#     This customized mapper produces two augmented images from a single image
#     instance. This mapper makes sure that the two augmented images have the same
#     cropping and thus the same size.

#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by the model.

#     This is the default callable to be used to map your dataset dict into training data.
#     You may need to follow it to implement your own one for customized logic,
#     such as a different way to read or transform images.
#     See :doc:`/tutorials/data_loading` for details.

#     The callable currently does the following:

#     1. Read the image from "file_name"
#     2. Applies cropping/geometric transforms to the image and annotations
#     3. Prepare data and annotations to Tensor and :class:`Instances`
#     """

#     def __init__(self, cfg, is_train=True, keep_tf_data=True, use_w=False):
#         self.augmentation = augs_with_transformer_patch(cfg, is_train, use_w=use_w)
#         # include crop into self.augmentation
#         if cfg.INPUT.CROP.ENABLED and False:
#             if cfg.INPUT.PAD_CROP:
#                 self.augmentation.insert(
#                     0, RandomCropAndPad(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
#                 )
#             else:
#                 self.augmentation.insert(
#                     0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
#                 )
#             logging.getLogger(__name__).info(
#                 "Cropping used in training: " + str(self.augmentation[0])
#             )
#             self.compute_tight_boxes = True
#         else:
#             self.compute_tight_boxes = False

#         # fmt: off
#         self.img_format = cfg.INPUT.FORMAT
#         self.mask_on = cfg.MODEL.MASK_ON
#         self.mask_format = cfg.INPUT.MASK_FORMAT
#         self.keypoint_on = cfg.MODEL.KEYPOINT_ON

#         self.is_train = is_train
#         self.keep_tf_data = keep_tf_data

#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         # utils.check_image_size(dataset_dict, image)

#         if "sem_seg_file_name" in dataset_dict:
#             sem_seg_gt = utils.read_image(
#                 dataset_dict.pop("sem_seg_file_name"), "L"
#             ).squeeze(2)
#         else:
#             sem_seg_gt = None

#         aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
#         transforms = aug_input.apply_augmentations(self.augmentation)
#         image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
#         image_shape = image_weak_aug.shape[:2]  # h, w

#         if 'foggy' in dataset_dict['file_name']:
#             mask_format = "bitmask"
#         elif 'cityscapes' in dataset_dict['file_name']:
#             mask_format = "bitmask"
#         elif 'acdc' in dataset_dict['file_name']:
#             mask_format = 'bitmask'
#         else:
#             mask_format = self.mask_format

#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))


#         if "annotations" in dataset_dict:
#             for anno in dataset_dict["annotations"]:
#                 # if not self.mask_on:
#                 #     anno.pop("segmentation", None)
#                 if not self.keypoint_on:
#                     anno.pop("keypoints", None)

#             annos = [
#                 utils.transform_instance_annotations(
#                     obj,
#                     transforms,
#                     image_shape,
#                     keypoint_hflip_indices=None,
#                 )
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             if len(annos):
#                 if 'segmentation' in annos[0].keys() and mask_format == "polygon":
#                     for lv1 in range(len(annos)):
#                         # annos[lv1]['segmentation'] = [annos[lv1]['segmentation'][0].reshape(-1,2)]
#                         annos[lv1]['segmentation'] = [x.tolist() for x in annos[lv1]['segmentation']]
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=mask_format
#             )

#             if self.compute_tight_boxes and instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

#             bboxes_d2_format = utils.filter_empty_instances(instances)
#             dataset_dict["instances"] = bboxes_d2_format

#         if self.keep_tf_data:
#             dataset_dict['tf_data'] = transforms

#         dataset_dict["image"] = torch.as_tensor(
#             np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
#         )
        
#         return dataset_dict

# class DatasetMapperWithWeakAugs(DatasetMapper):
#     """
#     This customized mapper produces two augmented images from a single image
#     instance. This mapper makes sure that the two augmented images have the same
#     cropping and thus the same size.

#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by the model.

#     This is the default callable to be used to map your dataset dict into training data.
#     You may need to follow it to implement your own one for customized logic,
#     such as a different way to read or transform images.
#     See :doc:`/tutorials/data_loading` for details.

#     The callable currently does the following:

#     1. Read the image from "file_name"
#     2. Applies cropping/geometric transforms to the image and annotations
#     3. Prepare data and annotations to Tensor and :class:`Instances`
#     """

#     def __init__(self, cfg, is_train=True):
#         self.augmentation = utils.build_augmentation(cfg, is_train)
#         # include crop into self.augmentation
#         if cfg.INPUT.CROP.ENABLED and is_train:
#             self.augmentation.insert(
#                 0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
#             )
#             logging.getLogger(__name__).info(
#                 "Cropping used in training: " + str(self.augmentation[0])
#             )
#             self.compute_tight_boxes = True
#         else:
#             self.compute_tight_boxes = False
#         self.strong_augmentation = build_strong_augmentation(cfg, is_train)

#         # fmt: off
#         self.img_format = cfg.INPUT.FORMAT
#         self.mask_on = cfg.MODEL.MASK_ON
#         self.mask_format = cfg.INPUT.MASK_FORMAT
#         self.keypoint_on = cfg.MODEL.KEYPOINT_ON
#         self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
#         # fmt: on
#         if self.keypoint_on and is_train:
#             self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
#                 cfg.DATASETS.TRAIN
#             )
#         else:
#             self.keypoint_hflip_indices = None

#         if self.load_proposals:
#             self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
#             self.proposal_topk = (
#                 cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
#                 if is_train
#                 else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
#             )
#         self.is_train = is_train

#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         # utils.check_image_size(dataset_dict, image)

#         if "sem_seg_file_name" in dataset_dict:
#             sem_seg_gt = utils.read_image(
#                 dataset_dict.pop("sem_seg_file_name"), "L"
#             ).squeeze(2)
#         else:
#             sem_seg_gt = None

#         aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
#         transforms = aug_input.apply_augmentations(self.augmentation)
#         image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
#         image_shape = image_weak_aug.shape[:2]  # h, w

#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

#         if self.load_proposals:
#             utils.transform_proposals(
#                 dataset_dict,
#                 image_shape,
#                 transforms,
#                 proposal_topk=self.proposal_topk,
#                 min_box_size=self.proposal_min_box_size,
#             )

#         if not self.is_train:
#             dataset_dict.pop("annotations", None)
#             dataset_dict.pop("sem_seg_file_name", None)
#             return dataset_dict

#         if "annotations" in dataset_dict:
#             for anno in dataset_dict["annotations"]:
#                 if not self.mask_on:
#                     anno.pop("segmentation", None)
#                 if not self.keypoint_on:
#                     anno.pop("keypoints", None)

#             annos = [
#                 utils.transform_instance_annotations(
#                     obj,
#                     transforms,
#                     image_shape,
#                     keypoint_hflip_indices=self.keypoint_hflip_indices,
#                 )
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=self.mask_format
#             )

#             if self.compute_tight_boxes and instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

#             bboxes_d2_format = utils.filter_empty_instances(instances)
#             dataset_dict["instances"] = bboxes_d2_format

#         dataset_dict["image"] = torch.as_tensor(
#             np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
#         )

#         return dataset_dict
    

# class DatasetMapperWithStrongAugs(DatasetMapper):
#     """
#     This customized mapper produces two augmented images from a single image
#     instance. This mapper makes sure that the two augmented images have the same
#     cropping and thus the same size.

#     A callable which takes a dataset dict in Detectron2 Dataset format,
#     and map it into a format used by the model.

#     This is the default callable to be used to map your dataset dict into training data.
#     You may need to follow it to implement your own one for customized logic,
#     such as a different way to read or transform images.
#     See :doc:`/tutorials/data_loading` for details.

#     The callable currently does the following:

#     1. Read the image from "file_name"
#     2. Applies cropping/geometric transforms to the image and annotations
#     3. Prepare data and annotations to Tensor and :class:`Instances`
#     """

#     def __init__(self, cfg, is_train=True):
#         self.augmentation = utils.build_augmentation(cfg, is_train)
#         # include crop into self.augmentation
#         if cfg.INPUT.CROP.ENABLED and is_train:
#             self.augmentation.insert(
#                 0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
#             )
#             logging.getLogger(__name__).info(
#                 "Cropping used in training: " + str(self.augmentation[0])
#             )
#             self.compute_tight_boxes = True
#         else:
#             self.compute_tight_boxes = False
#         self.strong_augmentation = build_strong_augmentation(cfg, is_train)

#         # fmt: off
#         self.img_format = cfg.INPUT.FORMAT
#         self.mask_on = cfg.MODEL.MASK_ON
#         self.mask_format = cfg.INPUT.MASK_FORMAT
#         self.keypoint_on = cfg.MODEL.KEYPOINT_ON
#         self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
#         # fmt: on
#         if self.keypoint_on and is_train:
#             self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
#                 cfg.DATASETS.TRAIN
#             )
#         else:
#             self.keypoint_hflip_indices = None

#         if self.load_proposals:
#             self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
#             self.proposal_topk = (
#                 cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
#                 if is_train
#                 else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
#             )
#         self.is_train = is_train

#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#         image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
#         # utils.check_image_size(dataset_dict, image)

#         if "sem_seg_file_name" in dataset_dict:
#             sem_seg_gt = utils.read_image(
#                 dataset_dict.pop("sem_seg_file_name"), "L"
#             ).squeeze(2)
#         else:
#             sem_seg_gt = None

#         aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
#         transforms = aug_input.apply_augmentations(self.augmentation)
#         image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
#         image_shape = image_weak_aug.shape[:2]  # h, w

#         if sem_seg_gt is not None:
#             dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

#         if self.load_proposals:
#             utils.transform_proposals(
#                 dataset_dict,
#                 image_shape,
#                 transforms,
#                 proposal_topk=self.proposal_topk,
#                 min_box_size=self.proposal_min_box_size,
#             )

#         if not self.is_train:
#             dataset_dict.pop("annotations", None)
#             dataset_dict.pop("sem_seg_file_name", None)
#             return dataset_dict

#         if "annotations" in dataset_dict:
#             for anno in dataset_dict["annotations"]:
#                 if not self.mask_on:
#                     anno.pop("segmentation", None)
#                 if not self.keypoint_on:
#                     anno.pop("keypoints", None)

#             annos = [
#                 utils.transform_instance_annotations(
#                     obj,
#                     transforms,
#                     image_shape,
#                     keypoint_hflip_indices=self.keypoint_hflip_indices,
#                 )
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=self.mask_format
#             )

#             if self.compute_tight_boxes and instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

#             bboxes_d2_format = utils.filter_empty_instances(instances)
#             dataset_dict["instances"] = bboxes_d2_format

#         # apply strong augmentation
#         # We use torchvision augmentation, which is not compatiable with
#         # detectron2, which use numpy format for images. Thus, we need to
#         # convert to PIL format first.
#         image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
#         image_strong_aug = np.array(self.strong_augmentation(image_pil))
#         dataset_dict["image"] = torch.as_tensor(
#             np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
#         )

#         return dataset_dict
    
def augs_with_transformer_patch(cfg, is_train, use_w=False):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    dino_patch = cfg.SEMISUPNET.DINO_PATCH_SIZE
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        sample_style = "choice"
    augmentation = [ResizeTransformDinoScale(min_size, dino_patch, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation

class ResizeTransformDinoScale(Transform):
    """
    Resize the image to a target size that is a multiple of the patch size
    """

    def __init__(self, new_h, dino_patch, inv=False, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        if type(new_h) == tuple:
            new_h = new_h[0]
        self.set_h = new_h
        self.dino_patch = dino_patch
        self.inv = inv
        self.interp = interp

    def apply_image(self, img, interp=None):
        # assert img.shape[:2] == (self.h, self.w)
        assert img.shape[0] < img.shape[1]
        assert len(img.shape) <= 4
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.new_h = floor(self.set_h / self.dino_patch) * self.dino_patch
        scale = self.new_h / self.h
        temp_w = scale * self.w
        self.new_w = round(temp_w / self.dino_patch) * self.dino_patch

        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransformDinoScale(self.new_h, self.new_w, self.h, self.w, self.interp)

# class ResizeTransformDinoScaleW(Transform):
#     """
#     Resize the image to a target size that is a multiple of the patch size
#     """

#     def __init__(self, new_w, dino_patch, inv=False, interp=None):
#         """
#         Args:
#             h, w (int): original image size
#             new_w (int): new image size
#             interp: PIL interpolation methods, defaults to bilinear.
#         """
#         # TODO decide on PIL vs opencv
#         super().__init__()
#         if interp is None:
#             interp = Image.BILINEAR
#         if type(new_w) == tuple:
#             new_w = new_w[1]
#         self.set_w = new_w
#         self.dino_patch = dino_patch
#         self.inv = inv
#         self.interp = interp

#     def apply_image(self, img, interp=None):
#         # assert img.shape[:2] == (self.h, self.w)
#         assert img.shape[0] < img.shape[1]
#         assert len(img.shape) <= 4
#         self.h = img.shape[0]
#         self.w = img.shape[1]
#         self.new_w = floor(self.set_w / self.dino_patch) * self.dino_patch
#         scale = self.new_w / self.w
#         temp_h = scale * self.h
#         self.new_h = round(temp_h / self.dino_patch) * self.dino_patch

#         interp_method = interp if interp is not None else self.interp

#         if img.dtype == np.uint8:
#             if len(img.shape) > 2 and img.shape[2] == 1:
#                 pil_image = Image.fromarray(img[:, :, 0], mode="L")
#             else:
#                 pil_image = Image.fromarray(img)
#             pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
#             ret = np.asarray(pil_image)
#             if len(img.shape) > 2 and img.shape[2] == 1:
#                 ret = np.expand_dims(ret, -1)
#         else:
#             # PIL only supports uint8
#             if any(x < 0 for x in img.strides):
#                 img = np.ascontiguousarray(img)
#             img = torch.from_numpy(img)
#             shape = list(img.shape)
#             shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
#             img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
#             _PIL_RESIZE_TO_INTERPOLATE_MODE = {
#                 Image.NEAREST: "nearest",
#                 Image.BILINEAR: "bilinear",
#                 Image.BICUBIC: "bicubic",
#             }
#             mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
#             align_corners = None if mode == "nearest" else False
#             img = F.interpolate(
#                 img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
#             )
#             shape[:2] = (self.new_h, self.new_w)
#             ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

#         return ret

#     def apply_coords(self, coords):
#         coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
#         coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
#         return coords

#     def apply_segmentation(self, segmentation):
#         segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
#         return segmentation

#     def inverse(self):
#         return ResizeTransformDinoScaleW(self.new_h, self.new_w, self.h, self.w, self.interp)

class RandomCropAndPad(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).

          Resize to original size with padding
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        crop_tf = CropTransform(w0, h0, cropw, croph)
        dh = h - croph
        dw = w - cropw

        # Only difference
        pad_tf = PadTransform(0,0,dw,dh)
        return TransformList([crop_tf, pad_tf])

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))
