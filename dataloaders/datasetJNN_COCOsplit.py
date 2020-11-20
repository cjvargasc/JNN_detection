import random
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset

from config import Config
from utils.utils import augment_img

import time


class DatasetJNN_COCOsplit(Dataset):

    def __init__(self, COCO_path, split_idx=0, is_training=True):

        self.is_training = is_training
        self.split_idx = split_idx

        if is_training:
            self.COCO_path = COCO_path + "train2017/"
            self.coco_dataset = dset.CocoDetection(root=self.COCO_path,
                                                   annFile=COCO_path + "annotations/instances_train2017.json")
        else:
            self.COCO_path = COCO_path + "val2017/"
            self.coco_dataset = dset.CocoDetection(root=self.COCO_path ,
                                                   annFile=COCO_path + "annotations/instances_val2017.json")

        # define splits
        split1 = ['person', 'airplane', 'boat', 'parking meter', 'dog', 'elephant', 'backpack', 'suitcase',
                  'sports ball', 'skateboard', 'wine glass', 'spoon', 'sandwich', 'hot dog', 'chair', 'dining table',
                  'mouse', 'microwave', 'refrigerator', 'scissors']
        split2 = ['bicycle', 'bus', 'traffic light', 'bench', 'horse', 'bear', 'umbrella', 'frisbee', 'kite',
                  'surfboard', 'cup', 'bowl', 'orange', 'pizza', 'couch', 'toilet', 'remote', 'oven',
                  'book', 'teddy bear']
        split3 = ['car', 'train', 'fire hydrant', 'bird', 'sheep', 'zebra', 'handbag', 'skis', 'baseball bat',
                  'tennis racket', 'fork', 'banana', 'broccoli', 'donut', 'potted plant', 'tv', 'keyboard',
                  'toaster', 'clock', 'hair drier']
        split4 = ['motorcycle', 'truck', 'stop sign', 'cat', 'cow', 'giraffe', 'tie', 'snowboard', 'baseball glove',
                  'bottle', 'knife', 'apple', 'carrot', 'cake', 'bed', 'laptop', 'cell phone','sink', 'vase',
                  'toothbrush']

        if split_idx == 1:
            self.split = split1
        elif split_idx == 2:
            self.split = split2
        elif split_idx == 3:
            self.split = split3
        elif split_idx == 4:
            self.split = split4
        else:
            print("invalid split selection, selecting default...")
            self.split = split1

        # compute split size
        imgs_list = np.array([])
        for cat in self.coco_dataset.coco.cats:
            if (self.is_training and self.coco_dataset.coco.cats[cat]['name'] not in self.split) or \
                    ((not self.is_training) and self.coco_dataset.coco.cats[cat]['name'] in self.split):
                imgs_list = np.concatenate((imgs_list, np.array(self.coco_dataset.coco.catToImgs[cat])))

        imgs_list = np.unique(imgs_list)
        self.dataset_len = imgs_list.size

        if not is_training:
           random.seed(123)

    def __getitem__(self, qindex):

        while True:
            try:
                cindex = random.randint(0, len(self.coco_dataset.coco.cats) - 1)
                random_class = self.coco_dataset.coco.cats[cindex]['name']
            except KeyError:
                continue
            if (self.is_training and not (random_class in self.split)) or (not self.is_training and random_class in self.split):
                break

        # get query data
        cat_img_set = self.coco_dataset.coco.catToImgs[cindex]
        qimg_id = cat_img_set[random.randint(0, len(cat_img_set) - 1)]
        qindex = self.coco_dataset.ids.index(qimg_id)

        # im_filename = self.coco_dataset.coco.imgs[qimg_id]['file_name']
        # q_im = Image.open(self.COCO_path + im_filename)
        # target = self.coco_dataset.coco.imgToAnns[qimg_id]
        q_im, target = self.coco_dataset[qindex]  # doing it manually seems to be faster

        qboxes = []
        qcats = []
        for annotation in target:
            qboxes.append(annotation['bbox'])
            qcats.append(self.coco_dataset.coco.cats[annotation['category_id']]['name'])
        qboxes, qcats = self.filter_boxes(qboxes, qcats, random_class)

        # Select a random box in the image as a query and crop
        query_random_index = random.randrange(0, len(qboxes))
        qbox = qboxes[query_random_index]
        # convert [x, y, w, h] to [x1, y1, x2, y2]
        qbox = [qbox[0], qbox[1], qbox[0] + qbox[2], qbox[1] + qbox[3]]
        qcat = qcats[query_random_index]

        # get target data
        cat_img_set = self.coco_dataset.coco.catToImgs[cindex]
        timg_id = cat_img_set[random.randint(0, len(cat_img_set) - 1)]
        tindex = self.coco_dataset.ids.index(timg_id)

        # im_filename = self.coco_dataset.coco.imgs[timg_id]['file_name']
        # t_im = Image.open(self.COCO_path + im_filename)
        # target = self.coco_dataset.coco.imgToAnns[qimg_id]
        t_im, target = self.coco_dataset[tindex]

        tboxes = []
        tcats = []
        for annotation in target:
            tboxes.append(annotation['bbox'])
            tcats.append(self.coco_dataset.coco.cats[annotation['category_id']]['name'])
        tboxes, tcats = self.filter_boxes(tboxes, tcats, qcat)

        # convert [x, y, w, h] to [x1, y1, x2, y2]
        boxes = []
        for box in tboxes:
            newbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            boxes.append(newbox)
        q_im = q_im.crop((qbox[0], qbox[1], qbox[2], qbox[3]))

        boxes = np.asarray(boxes, dtype=np.float32)

        # if q_im.mode == 'L':  # uncomment when loading manually
        #    q_im = q_im.convert('RGB')
        # if t_im.mode == 'L':
        #    t_im = t_im.convert('RGB')

        if self.is_training:

            t_im, boxes = augment_img(t_im, boxes)

            w, h = t_im.size[0], t_im.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize images
            q_im = q_im.resize((Config.imq_w, Config.imq_h))
            t_im = t_im.resize((Config.im_w, Config.im_h))

            # To float tensors
            q_im = torch.from_numpy(np.array(q_im)).float() / 255
            t_im = torch.from_numpy(np.array(t_im)).float() / 255
            q_im = q_im.permute(2, 0, 1)
            t_im = t_im.permute(2, 0, 1)

            boxes = torch.from_numpy(boxes)
            num_obj = torch.Tensor([boxes.size(0)]).long()

            return q_im, t_im, boxes, num_obj

        else:
            w, h = t_im.size[0], t_im.size[1]

            # resize images
            q_im = q_im.resize((Config.imq_w, Config.imq_h))
            t_im = t_im.resize((Config.im_w, Config.im_h))

            # To float tensors
            q_im = torch.from_numpy(np.array(q_im)).float() / 255
            t_im = torch.from_numpy(np.array(t_im)).float() / 255
            q_im = q_im.permute(2, 0, 1)
            t_im = t_im.permute(2, 0, 1)

            boxes = torch.from_numpy(boxes)

            return q_im, t_im, boxes, qcat, (w, h, self.coco_dataset.coco.imgs[qimg_id]['file_name'],
                                             self.coco_dataset.coco.imgs[timg_id]['file_name'])

    def filter_boxes(self, boxes, classes, qclass=''):
        # filters testing classes and categories different from query (qclass)
        out_boxes = []
        out_classes = []

        for i in range(len(classes)):
            if (self.is_training and classes[i] not in self.split) \
                    or (not self.is_training and classes[i] in self.split):
                if qclass == '' or (qclass != '' and classes[i] == qclass):
                    out_classes.append(classes[i])
                    out_boxes.append(boxes[i])

        return out_boxes, out_classes

    def __len__(self):
        return self.dataset_len


