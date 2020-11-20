import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset

import os
import cv2
import numpy as np
from PIL import Image

from config import Config
from model.darkJNN import DarkJNN
from model.decoder import decode
from dataloaders.datasetJNN import DatasetJNN
from dataloaders.datasetJNN_VOC import DatasetJNN_VOC
from dataloaders.datasetJNN_COCO import DatasetJNN_COCO
from dataloaders.datasetJNN_COCOsplit import DatasetJNN_COCOsplit


class Tester:

    @staticmethod
    def test():

        print("testing...")

        Config.batch_size = 1

        #Config.model_path = "testmodel_last.pt"
        Config.model_path = "/home/mmv/Documents/2.projects/JNN_detection/trained_models/dJNN_COCOsplit4/testmodel_last_split4.pt"
        print("mAP files output path: " + Config.mAP_path)

        model_path = Config.model_path

        print("model: ", model_path)
        print("conf: ", Config.conf_thresh)
        print("iou thresh:  ", Config.conf_thresh)

        if Config.dataset == "VOC":
            print("dataset: ", Config.voc_dataset_dir)
            dataset = DatasetJNN_VOC(Config.voc_dataset_dir, mode="test", year="2007", is_training=False)
        elif Config.dataset == "coco":
            print("dataset: ", Config.coco_dataset_dir)
            dataset = DatasetJNN_COCO(Config.coco_dataset_dir, is_training=False)
        elif Config.dataset == "coco_split":
            print("dataset: ", Config.coco_dataset_dir, "--Split: ", Config.coco_split)
            dataset = DatasetJNN_COCOsplit(Config.coco_dataset_dir, Config.coco_split, is_training=False)
        else:
            print("dataset: ", Config.testing_dir)
            folder_dataset = dset.ImageFolder(root=Config.testing_dir)
            dataset = DatasetJNN(imageFolderDataset=folder_dataset, is_training=False)

        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)

        model = DarkJNN()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        print("epoch: ", str(checkpoint['epoch'] + 1))

        model.cuda()
        model.eval()

        with torch.no_grad():

            for i, data in enumerate(dataloader, 0):

                if (i % 1000 == 0):
                    print(str(i) + "/" + str(len(dataset)))  # progress

                img0, img1, targets, label, im_infos = data
                img0, img1, targets = Variable(img0).cuda(), Variable(img1).cuda(), targets.cuda()

                model_output = model(img0, img1, targets)

                im_info = {'width': im_infos[0].item(), 'height': im_infos[1].item()}
                output = [item[0].data for item in model_output]

                detections = decode(output, im_info, conf_threshold=Config.conf_thresh,nms_threshold=Config.nms_thresh)

                if len(detections) > 0:

                    # mAP files
                    pair_id = im_infos[2][0].split('.')[0] + "_" +im_infos[3][0].split('.')[0]

                    detection_str = ""
                    gt_str = ""

                    f = open(Config.mAP_path + "groundtruths/" + pair_id + ".txt", "a+")
                    for box_idx in range(len(targets)):

                        gt_str += label[0].replace(" ", "_") + " " \
                                  + str(targets[0][box_idx][0].item()) + " " \
                                  + str(targets[0][box_idx][1].item()) + " " \
                                  + str(targets[0][box_idx][2].item()) + " " \
                                  + str(targets[0][box_idx][3].item()) + "\n"

                    f.seek(0)
                    if not (gt_str in f.readlines()):
                        f.write(gt_str)
                    f.close()

                    f = open(Config.mAP_path + "detections/" + pair_id + ".txt", "a+")
                    for detection in detections:
                        detection_str += label[0].replace(" ", "_") + " " \
                                      + str(detection[4].item()) + " "\
                                      + str(detection[0].item()) + " "\
                                      + str(detection[1].item()) + " "\
                                      + str(detection[2].item()) + " "\
                                      + str(detection[3].item()) + "\n"

                    f.seek(0)
                    if not (detection_str in f.readlines()):
                        f.write(detection_str)
                    f.close()

    @staticmethod
    def test_one_OL():
        """ Tests a a pair of images """

        print("testing one image...")

        Config.model_path = "/home/mmv/Documents/2.projects/JNN_detection/trained_models/dJNN_COCOsplit2/testmodel_last_split2.pt"

        model_path = Config.model_path

        model = DarkJNN()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.cuda()
        model.eval()

        # (3m1, 3m6), (rbc1, rbc43), hp(33971473, 70609284), blizzard(1, 6), gen_electric(7, 31), warner(10, 18)
        # goodyear(13, 20), airhawk(12, 1), gap(34, 36), levis(14, 30)
        q_name = "000000008629"
        t_name = "000000209530"
        q_im = Image.open("/home/mmv/Documents/3.datasets/coco/val2017/" + q_name + ".jpg")
        t_im = Image.open("/home/mmv/Documents/3.datasets/coco/val2017/" + t_name + ".jpg")

        w, h = t_im.size[0], t_im.size[1]
        im_infos = (w, h, q_name, t_name)

        cv_im = np.array(t_im)
        cv_im = cv_im[:, :, ::-1].copy()

        q_im = q_im.resize((Config.imq_w, Config.imq_h))
        t_im = t_im.resize((Config.im_w, Config.im_h))

        # To float tensors
        q_im = torch.from_numpy(np.array(q_im)).float() / 255
        t_im = torch.from_numpy(np.array(t_im)).float() / 255
        img0 = q_im.permute(2, 0, 1)
        img1 = t_im.permute(2, 0, 1)
        img0 = torch.unsqueeze(img0, 0)
        img1 = torch.unsqueeze(img1, 0)

        with torch.no_grad():
#
            img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()

            model_output = model(img0, img1, [])

            im_info = {'width': im_infos[0], 'height': im_infos[1]}
            output = [item[0].data for item in model_output]

            detections = decode(output, im_info, conf_threshold=Config.conf_thresh, nms_threshold=Config.nms_thresh)

            if len(detections) > 0:

                for detection in detections:
                    start_pt = (int(detection[0].item()), int(detection[1].item()))
                    end_pt = (int(detection[2].item()), int(detection[3].item()))
                    image = cv2.rectangle(cv_im, start_pt, end_pt, (0, 255, 0), 3)
                    print(start_pt, end_pt)

                cv2.imshow("res", image)
                cv2.waitKey()

    @staticmethod
    def test_one_COCO():
        """ Tests a a pair of images """

        print("testing one image...")

        Config.model_path = "/home/mmv/Documents/2.projects/JNN_detection/trained_models/dJNN_COCOsplit2/testmodel_last_split2.pt"
        model_path = Config.model_path

        model = DarkJNN()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        model.eval()

        coco_dataset = dset.CocoDetection(Config.coco_dataset_dir,
                                          Config.coco_dataset_dir + "annotations/instances_val2017.json")

        # (3m1, 3m6), (rbc1, rbc43), hp(33971473, 70609284), blizzard(1, 6), gen_electric(7, 31), warner(10, 18)
        # goodyear(13, 20), airhawk(12, 1), gap(34, 36), levis(14, 30)
        q_name = "000000024144"
        t_name = "000000306700"
        q_im = Image.open("/home/mmv/Documents/3.datasets/coco/val2017/" + q_name + ".jpg")
        t_im = Image.open("/home/mmv/Documents/3.datasets/coco/val2017/" + t_name + ".jpg")

        # find image id and (first) annotation
        for id in coco_dataset.coco.imgs:
            if coco_dataset.coco.imgs[id]['file_name'] == q_name + ".jpg":
                break
        for ann_id in coco_dataset.coco.anns:
            if coco_dataset.coco.anns[ann_id]['image_id'] == id:
                print(coco_dataset.coco.anns[ann_id])
                break
        qbox = coco_dataset.coco.anns[ann_id]['bbox']
        qbox = [qbox[0], qbox[1], qbox[0] + qbox[2], qbox[1] + qbox[3]]
        q_im = q_im.crop((qbox[0], qbox[1], qbox[2], qbox[3]))

        w, h = t_im.size[0], t_im.size[1]
        im_infos = (w, h, q_name, t_name)

        qcv_im = np.array(q_im)
        qcv_im = qcv_im[:, :, ::-1].copy()
        cv_im = np.array(t_im)
        cv_im = cv_im[:, :, ::-1].copy()

        q_im = q_im.resize((Config.imq_w, Config.imq_h))
        t_im = t_im.resize((Config.im_w, Config.im_h))

        # To float tensors
        q_im = torch.from_numpy(np.array(q_im)).float() / 255
        t_im = torch.from_numpy(np.array(t_im)).float() / 255
        img0 = q_im.permute(2, 0, 1)
        img1 = t_im.permute(2, 0, 1)
        img0 = torch.unsqueeze(img0, 0)
        img1 = torch.unsqueeze(img1, 0)

        with torch.no_grad():
            #
            img0, img1 = Variable(img0).cuda(), Variable(img1).cuda()

            model_output = model(img0, img1, [])

            im_info = {'width': im_infos[0], 'height': im_infos[1]}
            output = [item[0].data for item in model_output]

            detections = decode(output, im_info, conf_threshold=Config.conf_thresh, nms_threshold=Config.nms_thresh)

            if len(detections) > 0:

                for detection in detections:
                    start_pt = (int(detection[0].item()), int(detection[1].item()))
                    end_pt = (int(detection[2].item()), int(detection[3].item()))
                    image = cv2.rectangle(cv_im, start_pt, end_pt, (0, 255, 0), 3)
                    print(start_pt, end_pt)

                cv2.imshow("q", qcv_im)
                cv2.imshow("res", image)
                cv2.waitKey()
            else:
                print("No detctions found")

