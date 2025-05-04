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
    ├── cityscapes/
        ├── gtFine/
            ├── train/
            ├── test/
            └── val/
        └── leftImg8bit/
            ├── train/
            ├── test/
            └── val/
   └── cityscapes_foggy/
        ├── gtFine/
            ├── train/
            ├── test/
            └── val/
        └── leftImg8bit/
            ├── train/
            ├── test/
            └── val/
└── weights/
    ├── vgg16_bn-6c64b313_converted.pth
    ├── R-50.pkl
    ├── dinov2_vitl14_pretrain.pth
    └── dinov2_vitlg4_pretrain.pth
```

## Pre-trained Weights
We use ImageNet pre-trained VGG16 from [link](https://drive.google.com/file/d/1wNIjtKiqdUINbTUVtzjSkJ14PpR2h8_i/view?usp=sharing), ImageNet pre-trained ResNet from Detectron2 [link](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) and the DINOv2 ViT weights from [link](https://github.com/facebookresearch/dinov2).


