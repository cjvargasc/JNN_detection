import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset

import cv2
import numpy as np
from PIL import Image

from config import Config
from model.darkJNN import DarkJNN
from model.decoder import decode
from dataloaders.datasetJNN import DatasetJNN


class Tester:

    @staticmethod
    def test():

        print("testing...")

        Config.batch_size = 1
        iou_thresh = 0.5

        print("mAP files output path: " + Config.mAP_path)

        model_path = Config.best_model_path

        print("model: ", model_path)
        print("conf: ", Config.conf_thresh)
        print("iou thresh:  ", iou_thresh)

        print("dataset: ", Config.training_dir)

        folder_dataset = dset.ImageFolder(root=Config.testing_dir)
        dataset = DatasetJNN(imageFolderDataset=folder_dataset, is_training=False)

        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)

        model = DarkJNN()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

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
                    im_id = im_infos[2][0].split('.')[0]

                    detection_str = ""
                    gt_str = ""

                    f = open(Config.mAP_path + "groundtruths/" + im_id + ".txt", "a+")
                    for box_idx in range(len(targets)):

                        gt_str += label[0] + " " \
                                  + str(targets[0][box_idx][0].item()) + " " \
                                  + str(targets[0][box_idx][1].item()) + " " \
                                  + str(targets[0][box_idx][2].item()) + " " \
                                  + str(targets[0][box_idx][3].item()) + "\n"
                        if not (gt_str in f.readlines()):
                            f.write(gt_str)

                    f.close()

                    f = open(Config.mAP_path + "detections/" + im_id + ".txt", "a+")
                    for detection in detections:
                        detection_str += label[0] + " " \
                                      + str(detection[4].item()) + " "\
                                      + str(detection[0].item()) + " "\
                                      + str(detection[1].item()) + " "\
                                      + str(detection[2].item()) + " "\
                                      + str(detection[3].item()) + "\n"
                        if not (detection_str in f.readlines()):
                            f.write(detection_str)

                    f.close()

    @staticmethod
    def test_one():

        print("testing one image...")
        """
        Config.batch_size = 1
        conf_threshs = 0.3
        iou_thresh = 0.5

        model_path = Config.best_model_path
        im_path = "path_to_img"

        print("model: ", model_path)
        print("conf: ", conf_threshs)
        print("iou thresh:  ", iou_thresh)
        print("im_path: ", im_path)

        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor']

        pil_im = Image.open(im_path)

        cv_im = np.array(pil_im)
        cv_im = cv_im[:, :, ::-1].copy()

        im_infos = torch.FloatTensor([pil_im.size[0], pil_im.size[1]])
        pil_im = pil_im.resize((Config.im_w, Config.im_h))
        transform = transforms.Compose([transforms.ToTensor()])
        pil_im = transform(pil_im)
        pil_im = torch.unsqueeze(pil_im, 0).cuda()

        model = Yolov2()

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        model.cuda()
        model.eval()

        with torch.no_grad():

            im_data_variable = Variable(pil_im).cuda()

            yolo_outputs = model(im_data_variable)

            im_info = {'width': im_infos[0], 'height': im_infos[1]}
            output = [item[0].data for item in yolo_outputs]

            detections = yolo_eval(output, im_info, conf_threshold=conf_threshs,
                                   nms_threshold=Config.nms_thresh)

            if len(detections) > 0:

                for detection in detections:
                    cv_im = cv2.rectangle(cv_im, (detection[0], detection[1]),
                                          (detection[2], detection[3]), (0, 0, 255), 2)
                    cv_im = cv2.putText(cv_im, classes[int(detection[6].item())], (detection[0], detection[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("res", cv_im)
            cv2.waitKey()
            """
