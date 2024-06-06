import numpy as np
import cv2
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from itertools import groupby
import os


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class GT_COCO():
    def __init__(self, images_filepath, categories, path):
        self.categories = categories
        self.images_filepath = images_filepath
        self.path = path
        self.licenses = self.create_licenses()
        self.info = self.create_info()
        self.categories = self.create_categories()
        self.images, self.filename_to_id = self.create_images()
        self.annotations = self.create_annotations()
        self.create_json()
    def create_licenses (self):
        licenses = \
        [
            {
                "name": "",
                "id": 0,
                "url": ""
            }
        ]
        return licenses

    def create_info (self):
        info = \
        {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""
        }
        return info

    def create_categories(self):
        categories = dict()
        for i, category in enumerate(self.categories):
            categories['id'] = 1
            categories['name'] = category
            categories["supercategory"] = "car"
        return [categories]
    
    def create_images(self):
        images = []
        filename_to_id = {}
        for i, image_name in enumerate(sorted(os.listdir(self.images_filepath))):
            image = cv2.imread(self.images_filepath + '/' + image_name,0)
            data = \
            {
                "id": i + 1,
                "height": image.shape[0],
                "width": image.shape[1],
                "file_name": image_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            }
            filename_to_id[image_name] = i + 1
            images.append(data)
        return images, filename_to_id

    def create_annotations(self):
        annotations = []
        k = 1
        for image_name in sorted(os.listdir(self.images_filepath)):
            bbox = [0, 0, 0, 0]
            area = 0
            category_id = 1
            binary_mask = cv2.imread(self.images_filepath + '/' + image_name, 0)
            rle_seg = self.binary_mask_to_rle(binary_mask)
            image_id = self.filename_to_id[image_name]
            id = k
            k = k + 1
            data = \
             {
                "id": id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle_seg,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {
                                "occluded": False
                              }
             }
            annotations.append(data)
        return annotations
    def binary_mask_to_rle(self,binary_mask):
      rle = {'counts': [], 'size': list(binary_mask.shape)}
      counts = rle.get('counts')
      for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
          if i == 0 and value == 1:
              counts.append(0)
          counts.append(len(list(elements)))
      return rle
    
    def create_json(self):
        data = \
        {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations
        }
        print("saving json: ")
        with open(os.path.join('', self.path), 'w') as f:
            json.dump(data, f, cls = NpEncoder)

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

if __name__ == '__main__':

    #gt_coco = GT_COCO('/home/lukavetoshkin/APTK/dataset/gt_sem_seg', categories = ['car'], path = 'gt_aptk.json')
    coco_gt = COCO('/home/lukavetoshkin/APTK/OneFormer/gt_aptk.json')

    preds_path = '/home/lukavetoshkin/APTK/Mask2Former/results/binary'
    annotations_coco = []
    #image_ids = []
    for image_name in sorted(os.listdir(preds_path)):
        image_id = filename_to_id(coco_gt, image_name)
        #image_ids.append(image_id)

        bbox = [0,0,0,0]
        area = 0
        category_id = 1
        score = 1
        binary_mask = cv2.imread(preds_path + '/' + image_name, 0)
        rle_seg = binary_mask_to_rle(binary_mask)
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

    with open('mask2former.json', 'w') as f:
        json.dump(annotations_coco, f,cls = NpEncoder)
    