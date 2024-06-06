import torch
from pycocotools.coco import COCO
import os
import numpy as np
from itertools import groupby
import json
from tqdm import tqdm
import cv2

def filename_to_id(coco, file_name):
    for item in coco.dataset['images']:
        if item['file_name'] == file_name:
            return item['id']
    return 'НЕТ ТАКОГО ФАЙЛА'
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
if __name__ == '__main__':

    preds_path_oneformer = '/home/lukavetoshkin/APTK/OneFormer/results/preds'
    preds_path_mask2former = '/home/lukavetoshkin/APTK/Mask2Former/results/preds'
    annotations_coco = []
    coco_gt = COCO('/home/lukavetoshkin/APTK/OneFormer/gt_aptk.json')
    for pred_name in tqdm(sorted(os.listdir(preds_path_oneformer))):
        pred_oneformer = torch.load(preds_path_oneformer + '/' + pred_name)
        pred_mask2former = torch.load(preds_path_mask2former + '/' + pred_name)
        
        preds = 0.3 * pred_oneformer + 0.7 * pred_mask2former
        preds = preds.argmax(dim=0)
        preds_np = preds.numpy()
        preds_np = np.where(preds_np == 13, 255, 0)
        cv2.imwrite('/home/lukavetoshkin/APTK/OneFormer/results/ensemble_binary/' + pred_name[:-3]+'.png',preds_np)
        image_id = filename_to_id(coco_gt, pred_name[:-3]+'.png')
        #image_ids.append(image_id)

        bbox = [0,0,0,0]
        area = 0
        category_id = 1
        score = 1
        rle_seg = binary_mask_to_rle(preds_np)
        data =   {
                  "image_id": image_id,
                  "category_id": category_id,
                  "segmentation": rle_seg,
                  "area": area,
                  "bbox": bbox,
                  "iscrowd": 0,
                  "score": score
              }
        annotations_coco.append(data)
    
    with open('ensemble.json', 'w') as f:
        json.dump(annotations_coco, f,cls = NpEncoder)