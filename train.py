import time
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

from dataloaders.datasetJNN import DatasetJNN
from dataloaders.datasetJNN_VOC import DatasetJNN_VOC
from dataloaders.datasetJNN_COCO import DatasetJNN_COCO
from dataloaders.datasetJNN_COCOsplit import DatasetJNN_COCOsplit
from model.darkJNN import DarkJNN
from utils.utils import Utils
from config import Config


class Trainer:

    @staticmethod
    def train():

        torch.cuda.manual_seed(123)

        print("Training process initialized...")

        if Config.dataset == "VOC":
            print("dataset: ", Config.voc_dataset_dir)
            dataset = DatasetJNN_VOC(Config.voc_dataset_dir)
        elif Config.dataset == "coco":
            print("dataset: ", Config.coco_dataset_dir)
            dataset = DatasetJNN_COCO(Config.coco_dataset_dir)
        elif Config.dataset == "coco_split":
            print("dataset: ", Config.coco_dataset_dir, "--Split: ", Config.coco_split)
            dataset = DatasetJNN_COCOsplit(Config.coco_dataset_dir, Config.coco_split)
        else:
            print("dataset: ", Config.training_dir)
            folder_dataset = dset.ImageFolder(root=Config.training_dir)
            dataset = DatasetJNN(imageFolderDataset=folder_dataset)

        train_dataloader = DataLoader(dataset,
                                      shuffle=True,
                                      num_workers=Config.num_workers,
                                      batch_size=Config.batch_size,
                                      drop_last=True,
                                      collate_fn=Utils.custom_collate_fn)

        print("lr:     ", Config.lr)
        print("batch:  ", Config.batch_size)
        print("epochs: ", Config.epochs)

        model = DarkJNN()

        lr = Config.lr

        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=Config.momentum, weight_decay=Config.weight_decay)

        starting_ep = 0

        if Config.continue_training:
            checkpoint = torch.load(Config.model_path)
            model.load_state_dict(checkpoint['model'])
            starting_ep = checkpoint['epoch'] + 1
            lr = checkpoint['lr']
            Trainer.adjust_learning_rate(optimizer, lr)
            print("starting epoch: ", starting_ep)

        model.cuda()
        model.train()

        counter = []
        loss_history = []

        best_loss = 10 ** 15
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement

        for epoch in range(starting_ep, Config.epochs):

            start_time = time.time()

            average_model_time = 0
            average_optim_time = 0

            average_epoch_loss = 0
            average_loc_loss = 0
            average_conf_loss = 0

            if epoch in Config.decay_lrs:
                lr = Config.decay_lrs[epoch]
                Trainer.adjust_learning_rate(optimizer, lr)
                print('adjust learning rate to {}'.format(lr))

            for i, data in enumerate(train_dataloader, 0):

                #if (i % 3000 == 0):
                #    print(str(i) + "/" + str(len(train_dataloader)))  # progress

                img0, img1, targets, num_obj = data
                img0, img1, targets, num_obj = Variable(img0).cuda(), Variable(img1).cuda(), targets.cuda(), num_obj.cuda()

                model_timer = time.time()
                loc_l, conf_l = model(img0, img1, targets, num_obj, training=True)
                loss = loc_l.mean() + conf_l.mean()
                model_timer = time.time() - model_timer
                average_model_time += model_timer

                optim_timer = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optim_timer = time.time() - optim_timer
                average_optim_time += optim_timer

                average_epoch_loss += loss
                average_loc_loss += loc_l
                average_conf_loss += conf_l

            end_time = time.time() - start_time
            print("time: ", end_time)

            others_timer = end_time - average_model_time - average_optim_time
            print("data+ time: ", others_timer)
            print("model time: ", average_model_time)
            print("optim time: ", average_optim_time)

            average_epoch_loss = average_epoch_loss / i

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(epoch)
            loss_history.append(average_epoch_loss.item())

            if average_epoch_loss < best_loss:
                print("------Best:")
                break_counter = 0
                best_loss = average_epoch_loss
                best_epoch = epoch
                save_name = Config.best_model_path
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': average_epoch_loss,
                    'lr': lr
                }, save_name)

            save_name = Config.model_path
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': average_epoch_loss,
                'lr': lr
            }, save_name)

            print("")
            if break_counter >= 20:
                print("Training break...")
                #break

            break_counter += 1

        print("best: ", best_epoch)
        plt.plot(counter, loss_history)
        plt.show()

    @staticmethod
    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
