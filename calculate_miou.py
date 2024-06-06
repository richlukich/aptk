import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

if __name__ == '__main__':

    
    miou = 0
    ensemble = True
    if not ensemble:
        preds_path = '/home/lukavetoshkin/APTK/OneFormer/results/binary' 
        for i, image_name in tqdm(enumerate(sorted(os.listdir(preds_path)))):
            binary_mask = cv2.imread(preds_path + '/' + image_name, 0)
            binary_mask_gt = cv2.imread('/home/lukavetoshkin/APTK/dataset/gt_sem_seg' + '/' + image_name, 0)

            iou = calculate_iou(binary_mask, binary_mask_gt)
            miou += iou
    else:
        preds_path_oneformer = '/home/lukavetoshkin/APTK/OneFormer/results/preds'
        preds_path_mask2former = '/home/lukavetoshkin/APTK/Mask2Former/results/preds'
        for i,pred_name in tqdm(enumerate(sorted(os.listdir(preds_path_oneformer)))):
            pred_oneformer = torch.load(preds_path_oneformer + '/' + pred_name)
            pred_mask2former = torch.load(preds_path_mask2former + '/' + pred_name)
            
            preds = 0.3 * pred_oneformer + 0.7 * pred_mask2former
            preds = preds.argmax(dim=0)
            preds_np = preds.numpy()
            preds_np = np.where(preds_np == 13, 255, 0)
            binary_mask_gt = cv2.imread('/home/lukavetoshkin/APTK/dataset/gt_sem_seg' + '/' + pred_name[:-3]+'.png', 0)
            iou = calculate_iou(preds_np, binary_mask_gt)
            miou += iou

    print ('mIoU = ', miou / (i+1))

