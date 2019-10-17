import os
import time
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.modified3dUNet import Modified3dUNet_SE
from loss import one_hot_embedding, weighted_categorical_dice, WeightedDiceLoss, LovaszLoss
from dataloader import get_loader
from utils import _create_optimizer, _create_scheduler


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)


class Network(object):
    def __init__(self, opt, data_name, data_type, class_weights, crop_range, hu_range, down_size, n_epochs, loss_change, milestones):
        self.data_dir = opt.data_dir
        self.checkpoint_dir = opt.checkpoint_dir
        self.lr = opt.lr
        self.n_splits = opt.n_splits

        self.n_classes = 4
        self.data_name = data_name
        self.data_type = data_type
        self.class_weights = class_weights
        self.crop_range = crop_range
        self.hu_range = hu_range
        self.down_size = down_size
        self.n_epochs = n_epochs
        self.milestones = list(milestones)
        self.loss_change = loss_change

        self.val_weights = (0,1/3,1/3,1/3) if self.data_name == 'CHD' else (0,1/2,1/4,1/4)
        self.best_val_dice = float('-inf')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.build_model()

        self.optimizer = _create_optimizer(opt, self.net)
        self.scheduler = _create_scheduler(self.optimizer, self.milestones)
        # self.scheduler = _create_scheduler(self.optimizer, patience)

    def build_model(self):
        self.net = Modified3dUNet_SE(n_classes=self.n_classes).to(self.device)
        self.net.apply(weights_init)

        total_params = sum(p.numel() for p in self.net.parameters())
        print('Total # of Parameters:', total_params)

    def build_dataloader(self, epoch):
        print(f'make dataloder epoch {epoch}')
        self.train_loader = get_loader(self.data_dir, self.data_name, self.data_type, self.crop_range, self.hu_range,
                                       self.down_size, num_workers=2, shuffle=True,mode='train', epoch=epoch,
                                       n_splits=self.n_splits)
        self.val_loader = get_loader(self.data_dir, self.data_name, self.data_type, self.crop_range, self.hu_range,
                                     self.down_size, num_workers=0, shuffle=False, mode='val', epoch=epoch,
                                     n_splits=self.n_splits)
        self.num_steps = len(self.train_loader)

    def train(self):

        start_time = time.time()
        for epoch in range(self.n_epochs):

            self.build_dataloader(epoch)

            # loss change
            if epoch < self.loss_change:
                criterion = WeightedDiceLoss(self.n_classes, self.class_weights).to(self.device)
            else:
                criterion = LovaszLoss().to(self.device)

            for step, (images_, labels_, _, _) in enumerate(self.train_loader):
                # Data shape: (N, C, D, H, W)
                images, labels = images_.to(self.device), labels_.to(self.device)

                preds = self.net(images)
                # images.shape (N, 1, D, H, W)
                # labels.shape (N, 1, D, H, W)
                # preds.shape (N, 4, D, H, W)

                preds = nn.Softmax(dim=1)(preds) # 이걸 잊었다!!!!!!!
                preds_flat = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
                labels_flat = labels.squeeze(1).view(-1).long()
                # preds_flat.shape (N*D*H*W, 4)
                # labels_flat.shape (N*D*H*W, 1)

                del images_, labels_, images, labels, preds
                torch.cuda.empty_cache()

                self.net.zero_grad()
                loss = criterion(preds_flat, labels_flat)
                loss.backward()
                self.optimizer.step()

                step_end_time = time.time()
                print('[%d/%d][%d/%d] - time_passed: %.2f, Loss: %.2f'
                      % (epoch, self.n_epochs, step, self.num_steps, step_end_time - start_time, loss))

                del preds_flat, labels_flat
                torch.cuda.empty_cache()

            # validation
            val_dice = self.validate()
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(self.net.state_dict(), os.path.join(self.checkpoint_dir, '%s.ckpt' % self.data_type))
                print('Val Dice Improved! Saved checkpoint: %s.ckpt' % self.data_type)

            # if epoch >= self.loss_change:
            self.scheduler.step()
            print('Learning rate: %f' % self.optimizer.param_groups[0]['lr'])

    def validate(self):
        top = self.crop_range[0]
        bottom = self.crop_range[1]
        left = self.crop_range[2]
        right = self.crop_range[3]

        val_dices = []

        for images_, labels_, orig_size, crop_size in self.val_loader:
            with torch.no_grad():
                # image만 downsize된게 들어오게 함.
                images, labels = images_.to(self.device), labels_.to(self.device)
                orig_size = tuple(torch.cat(orig_size).numpy())
                crop_size = tuple(torch.cat(crop_size).numpy())

                preds = self.net(images)

                preds = F.interpolate(preds, size=crop_size, mode='trilinear', align_corners=True)
                preds = torch.argmax(preds[0], 0)

                # added
                if orig_size[1] == 512:
                    zeros = torch.zeros(orig_size)
                    zeros[:,top:-bottom,left:-right] = preds
                    preds = zeros
                    del zeros

                preds_flat = preds.contiguous().view(-1).long()
                labels_flat = labels.squeeze().view(-1).long()

            dice = weighted_categorical_dice(one_hot_embedding(preds_flat, self.n_classes),
                                             one_hot_embedding(labels_flat, self.n_classes),
                                             self.n_classes, self.val_weights)
            val_dices.append(dice)

            del images_, labels_, images, labels, preds, preds_flat, labels_flat
            torch.cuda.empty_cache()

        val_dice = np.mean(val_dices)
        print('Weighted Dice Coefficient Score: %.4f' % val_dice)

        return val_dice

##################################################################################################################
##################################################################################################################
##################################################################################################################


class InferNetwork(object):
    def __init__(self, opt, data_name, data_type, crop_range, hu_range, down_size):
        self.data_dir = opt.data_dir
        self.checkpoint_dir = opt.checkpoint_dir
        self.output_dir = opt.output_dir

        self.n_classes = 4
        self.data_name = data_name
        self.data_type = data_type
        self.crop_range = crop_range
        self.hu_range = hu_range
        self.down_size = down_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.build_model()
        self.build_dataloader()

    def build_model(self):
        self.net = Modified3dUNet_SE(n_classes=self.n_classes).to(self.device)
        self.net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, '%s.ckpt' % self.data_type),
                                            map_location=self.device))
        print('Checkpoint Loaded: %s.ckpt' % self.data_type)

    def build_dataloader(self):
        self.test_loader, self.test_names = get_loader(self.data_dir, self.data_name, self.data_type, self.crop_range,
                                                       self.hu_range, self.down_size, num_workers=2, shuffle=False,
                                                       mode='test')

    def infer(self):
        top = self.crop_range[0]
        bottom = self.crop_range[1]
        left = self.crop_range[2]
        right = self.crop_range[3]

        for i, (image_, orig_size, crop_size) in enumerate(self.test_loader):
            with torch.no_grad():
                image = image_.to(self.device)
                orig_size = tuple(torch.cat(orig_size).numpy())
                crop_size = tuple(torch.cat(crop_size).numpy())

                pred = self.net(image) # (1, 4, 192, 256, 256)

                # argmax 먼저 하는 거 test
                # pred = torch.argmax(pred[0], 0).float()
                # pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=orig_size, mode='nearest')

                # 기존
                pred = F.interpolate(pred, size=crop_size, mode='trilinear', align_corners=True)
                pred = torch.argmax(pred[0], 0)  # interpolate와의 순서를 어떻게 해야 할까.

                # added
                if orig_size[1] == 512:
                    zeros = torch.zeros(orig_size)
                    zeros[:, top:-bottom, left:-right] = pred.squeeze()
                    pred = zeros
                    del zeros

                output = pred.squeeze().detach().cpu().numpy().astype(np.uint8)

            output = sitk.GetImageFromArray(output)

            if not os.path.exists(os.path.join(self.output_dir, self.data_name)):
                os.mkdir(os.path.join(self.output_dir, self.data_name))

            output_name = self.test_names[i].split('.')[0] + '_output.mha'
            sitk.WriteImage(output, os.path.join(self.output_dir, self.data_name, output_name))
            print('Saved output: %s' % output_name)

            del image_, image, pred, output
            torch.cuda.empty_cache()

# HVSMR 데이터는 image가 float64, 레이블이 int16이다.
# 아산데이터는 Uint16, int8
