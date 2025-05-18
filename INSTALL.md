# Installation

## Python environment versions

- python 3.10
- numpy 1.26.4
- torch 2.4.1+cu121
- (recommended) xformers 0.0.28.post1
- cityscapesscripts
- shapely 2.1.0
- detectron2 from source commit c69939a (later versions a different way to load cityscapesscripts)

## Our testing environment

- 2 RTX6000 (8 images for source, 8 images for target)


## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Data organization

We use folder organization like this:

```shell
dino_teacher/
├── datasets/
    ├── acdc/
        ├── gt_detection_trainval/gt_detection/
            ├── fog/
                ├── train/
                    └── img_anno
                ├── val/
                    └── img_anno
                ├── instancesonly_fog_test_images_info.json
                ├── instancesonly_fog_train_gt_detection.json
                └── instancesonly_fog_val_gt_detection.json
            ├── night/
            ├── rain/
            ├── snow/
            ├── instancesonly_test_images_info.json
            ├── instancesonly_train_gt_detection.json
            ├── instancesonly_val_gt_detection.json
            └── parse_acdc_annos.py
        └── rgb_anon_trainvaltest/
    ├── cityscapes/
        ├── gtFine/
            ├── train/
            ├── test/
            └── val/
        └── leftImg8bit/
            ├── train/
            ├── test/
            └── val/
   ├── cityscapes_foggy/
        ├── gtFine/
            ├── train/
            ├── test/
            └── val/
        └── leftImg8bit/
            ├── train/
            ├── test/
            └── val/
    ├── bdd/
        ├── images/
            ├── train/
            └── val/
        └── labels/
            ├── train/
                └── day/
                    └── img_annos.pkl
            ├── val/
                └── day/
                    └── img_annos.pkl
            ├── det_train.json
            ├── det_val.json
            └── parse_bdd_annos.py
    ├── vgg16_bn-6c64b313_converted.pth
    ├── R-50.pkl
    ├── dinov2_vitl14_pretrain.pth
    └── dinov2_vitlg4_pretrain.pth
```

## Building the datasets
- ACDC: move the `parse_acdc_annos.py` file inside the `gt_detection` directory and run it to generate the `img_anno` files for each split. Note that the rain split train annotation file `instancesonly_fog_train_gt_detection.json` contains the training annotations for all splits and so we include a corrected version [link](https://drive.google.com/file/d/1XskIoTf2eOgCJ3tONBE1BIO8Oac6P9u9/view?usp=drive_link). The parsed `img_anno` files are also available at [link](https://drive.google.com/drive/folders/1RfWRLnn8OX2JH44-e5ANPct4Qsdnl_zU?usp=drive_link).

- BDD100k: The parsed `img_annos.pkl` files are available at [link](https://drive.google.com/drive/folders/1GKRTjMer80ln2d_T_tkXAzaO0o-58j7y?usp=sharing). Alternatively, move the `parse_bdd_annos.py` inside the `labels` directory and run it to generate the `img_annos.pkl` files for each split. 

## Pre-trained Weights
We use ImageNet pre-trained VGG16 from [link](https://drive.google.com/file/d/1wNIjtKiqdUINbTUVtzjSkJ14PpR2h8_i/view?usp=sharing), ImageNet pre-trained ResNet from Detectron2 [link](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) and the DINOv2 ViT weights from [link](https://github.com/facebookresearch/dinov2).


