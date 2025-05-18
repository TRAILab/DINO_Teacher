# Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection

This is the PyTorch implementation of our CVPR 2025 paper: <br>
**Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection**<br>
 Marc-Antoine Lavoie, Anas Mahmoud, Steven Waslander<br>
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025 <br>
[[Paper](https://arxiv.org/pdf/2503.23220)]

`DINO Teacher` is a domain adaptive object detection method that leverages VFMs as a source of pseudo-labels and for cross-domain alignment. Our work is based off [`Adaptive Teacher`](https://github.com/facebookresearch/adaptive_teacher).

<p align="center">
<img src="schematic.png" width="85%">
</p>

## Installation
Please refer to [INSTALL.md](INSTALL.md) for the installation of `DINO Teacher`.

## Training
- Train the DINO labeller (you can replace the test datasets).
```shell
python train_net.py\
      --num-gpus 2\
      --config configs/vit_labeller.yaml\
      OUTPUT_DIR output/dino_label/test_vitl\
      SOLVER.IMG_PER_BATCH_LABEL 8\
      DATASETS.TEST '("cityscapes_val","cityscapes_foggy_val","BDD_day_val")'\
      SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14
```

- Generate the target domain pseudo-labels. Note that we evaluate on the train split (`DATASETS.TEST=("BDD_day_train",)`) to generate the train split pseudo-labels. We use the checkpoint resuming function, and so you should set the desired model by specifying the `OUTPUT_DIR` config variable and setting the desired checkpoint in the `last_checkpoint` file. The `SEMISUPNET.DINO_BBONE_MODEL` parameter initializes the ViT model and must match the size of the checkpoint for parameter loading. We evalate on a single GPU.
```shell
python train_net.py\
      --num-gpus 1\
      --resume\
      --gen-labels\
      --config configs/vit_labeller.yaml\
      OUTPUT_DIR output/dino_label/test_vitl\
      DATASETS.TEST '("BDD_day_train",)'\
      SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14
```

- Run `DINO Teacher` on the desired target domain. You may have to specify the correct path to the labeller annotations.
```shell
python train_net.py\
      --num-gpus 2\
      --resume\
      --config configs/vgg_city2bdd.yaml\
      SEMISUPNET.LABELER_TARGET_PSEUDOGT output/dino_label/test_vitl/predictions/BDD_day_train_dino_anno_vitl.pkl
```



# Results and Weights
## DINO ViT Labellers
The DINO labellers are all trained on the original Cityscapes only. All results are mAP@.50.

|     Backbone     | Cityscapes |  Foggy Cityscapes  |  BDD100k  |  Weights  | Forward Pass Labels |
|:-:               |:-:         |:-:                 |:-:        |:-:        |:-:                     | 
| ViT-L            |61.3        |54.6                |45.7       | [link](https://drive.google.com/file/d/1JOq4_uCjBl6nYEB__L54QiL5iXTvn6BA/view?usp=drive_link)          | [FCS](https://drive.google.com/file/d/1oc5LIU7OxhFI5Cu2Rm2Uw2mBfqmrIVyF/view?usp=drive_link), [BDD](https://drive.google.com/file/d/1zstBlg-IJWO9SQ2sDwD2GfPi_f2nTylO/view?usp=drive_link) |
| ViT-G            |64.3        |58.8                |51.1       | [link](https://drive.google.com/file/d/1nIfAmzIcTf_ZtYWCYDsBanufFM-bdAW4/view?usp=drive_link)          | [FCS](https://drive.google.com/file/d/1VVBWFVr57f8llQvI5pRD3qbxG7TZi93v/view?usp=drive_link), [BDD](https://drive.google.com/file/d/1u4evLs6jPbwanVahECJG52hivTFVht10/view?usp=drive_link) |

## Student Models
The student models are trained on the source Cityscapes with ground truth before using the DINO labellers pseudo-labels on the target domain.

| Target Domain    | Backbone | Labeller Size | Align. Teacher | mAP@.50 | Weights | 
|:-:               |:-:       |:-:            |:-:             |:-:      |:-:      |
| Foggy Cityscapes | VGG      | ViT-G         | ViT-B          | 55.4    | [link](https://drive.google.com/file/d/1LbJevSZgdMAay576ykcESXQUeoNjrLt6/view?usp=drive_link)    |
| BDD100k          | VGG      | ViT-G         | ViT-B          | 47.8    | [link]https://drive.google.com/file/d/1EG-ldsKT5VjEck3Ke0uAACwWEOoeWrJe/view?usp=drive_link)         |

## Citation
If you use DINO Teacher in your research, please consider citing:
```
@article{lavoie2025large,
  title={Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection},
  author={Lavoie, Marc-Antoine and Mahmoud, Anas and Waslander, Steven L},
  journal={arXiv preprint arXiv:2503.23220},
  year={2025}
}
```

## License
`DINO Teacher` is released under the [Apache 2.0 license](./LICENSE). 
