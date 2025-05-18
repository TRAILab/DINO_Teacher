import os
import pickle
import json
import numpy as np
import cv2
from detectron2.structures.boxes import BoxMode

splits = ['train','val']
cwd = os.getcwd()
for split in splits:
    print(f'Processing {split} set')
    anno_dir = cwd + '/' + split 
    img_dir = cwd.rsplit('/',1)[0] + '/images/' + split
    dataset_path = './datasets/bdd/images/' + split + '/'
    file_in = anno_dir + f'/det_{split}.json'
    with open(file_in, 'rb') as f_in:
        data_orig = json.load(f_in)

    # ID 5 corresponds to 'train', but we don't evaluate on that class
    cat2id = {'pedestrian':0, 'rider':1, 'car':2, 'truck':3, 'bus':4, 'motorcycle':6, 'bicycle':7}
    data_out = []
    for i,img_data in enumerate(data_orig):
        if img_data['attributes']['timeofday'] == 'daytime':
            img_name = img_data['name']
            annos_orig = img_data
            annos_new = []
            if 'labels' in img_data.keys():
                for anno in img_data['labels']:
                    if anno['category'] in cat2id.keys():
                        bbox = list(anno['box2d'].values())
                        bbox[2:] = [x-y for x,y in zip(bbox[2:],bbox[:2])]
                        anno_new = {'iscrowd': 0,
                                    'class_id': cat2id[anno['category']],
                                    'bbox': bbox,
                                    'area': bbox[2] * bbox[3],
                                    'segmentation': None,
                                    'image_id': i,
                                    'id': anno['id'],
                                    'bbox_mode': BoxMode.XYWH_ABS}
                        annos_new.append(anno_new)
            img_anno = {'file_name': dataset_path + img_name,
                        'height': 720,
                        'width': 1280,
                        'id': i,
                        'image_id': i,
                        'annotations': annos_new,}
            data_out.append(img_anno)

    file_out = anno_dir + f'/day/img_annos.pkl'
    with open(file_out, 'wb') as f_out:
        pickle.dump(data_out,f_out)

