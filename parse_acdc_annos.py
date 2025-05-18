import json
import os
from detectron2.structures import BoxMode
import pickle

dirs = next(os.walk('.'))[1]
for dir in dirs:
    curr_path = './' + dir
    gt_files = os.listdir(curr_path)
    for file_ in gt_files:
        if 'val' in file_ or 'train' in file_:
            f_in = '/'.join((curr_path,file_))
            if os.path.isdir(f_in):
                continue
            with open(f_in, 'r') as fin:
                data = json.load(fin)
            
            img_dict = {}
            for anno in data['annotations']:
                # del anno['segmentation']
                anno['bbox_mode'] = BoxMode.XYWH_ABS
                
                if anno['image_id'] in img_dict.keys():
                    img_dict[anno['image_id']].append(anno)
                else:
                    img_dict[anno['image_id']] = [anno]

            img_list = []
            for img in data['images']:
                curr_dict = img
                curr_dict['image_id'] = curr_dict['id']
                curr_dict['file_name'] = './datasets/acdc/rgb_anon_trainvaltest/rgb_anon/' + img['file_name']
                if curr_dict['id'] in img_dict.keys():
                    curr_dict['annotations'] = img_dict[curr_dict['id']]
                else:
                    curr_dict['annotations'] = []
                img_list.append(curr_dict)
            
            split = file_.split('_')[2]
            f_out = '/'.join((curr_path,split))
            os.makedirs(f_out, exist_ok=True)
            f_out += '/img_anno' 
            with open(f_out, 'wb') as fout:
                pickle.dump(img_list,fout)