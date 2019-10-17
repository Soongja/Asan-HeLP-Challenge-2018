import os
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import SimpleITK as sitk
from utils import k_folds


class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, data_name, data_type, crop_range, hu_range, down_size, mode, epoch, n_splits):
        self.folder = folder
        self.data_name = data_name
        self.data_type = data_type
        self.crop_range = crop_range
        self.hu_range = hu_range
        self.down_size = down_size
        self.mode = mode

        if not os.path.exists(self.folder):
            raise Exception("[!] {} does not exist.".format(self.folder))

        if self.mode in ['train', 'val']:
            self.image_names = os.listdir(os.path.join(self.folder, 'train', self.data_name, 'image'))
            self.label_names = os.listdir(os.path.join(self.folder, 'train', self.data_name, 'label'))
            self.image_names = [f for f in self.image_names if self.data_type in f]
            self.label_names = [f for f in self.label_names if self.data_type in f]
            self.image_names.sort()
            self.label_names.sort()

            # k-fold cross validation
            for i, (train_idx, val_idx) in enumerate(k_folds(n_splits=n_splits, subjects=len(self.image_names), frames=1)):
                if i == (epoch % n_splits):
                    print(f' i {i} train_idx {train_idx} / val_idx {val_idx}')
                    if self.mode == 'train':
                        self.image_names = np.array(self.image_names)[train_idx] # 그냥 list 안됨?
                        self.label_names = np.array(self.label_names)[train_idx]
                    else: # val
                        self.image_names = np.array(self.image_names)[val_idx]
                        self.label_names = np.array(self.label_names)[val_idx]

        else: # test
            self.image_names = os.listdir(os.path.join(self.folder, self.mode, self.data_name))
            self.image_names = [f for f in self.image_names if self.data_type in f]
            self.image_names.sort()

        print(self.mode, 'dataset size:', len(self.image_names))

        if len(self.image_names) == 0:
            raise Exception("[!] No images are found in {}".format(os.path.join(self.folder, self.phase, self.data_name)))

    def __getitem__(self, index):
        top = self.crop_range[0]
        bottom = self.crop_range[1]
        left = self.crop_range[2]
        right = self.crop_range[3]

        min_bound = self.hu_range[0]
        max_bound = self.hu_range[1]

        if self.mode in ['train', 'val']:
            image = sitk.ReadImage(os.path.join(self.folder, 'train', self.data_name, 'image', self.image_names[index]))
            label = sitk.ReadImage(os.path.join(self.folder, 'train', self.data_name, 'label', self.label_names[index]))

            # ------------------------- to array -------------------------
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            orig_size = image.shape

            # ------------------------- crop -------------------------
            if image.shape[1] == 512:
                image = image[:,top:-bottom,left:-right]
                if self.mode == 'train':
                    label = label[:,top:-bottom,left:-right]
            crop_size = image.shape

            # -------------------- clip and normalize(image only) --------------------
            image = (image - min_bound) / (max_bound - min_bound)
            image[image > 1] = 1
            image[image < 0] = 0

            image = (image * 2) - 1

            # ------------------------- to tensor -------------------------
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label).unsqueeze(0).float()

            # ------------------------- downsizing -------------------------
            image = F.interpolate(image.unsqueeze(0), size=self.down_size, mode='trilinear', align_corners=True).squeeze(0)
            if self.mode == 'train':
                label = F.interpolate(label.unsqueeze(0), size=self.down_size, mode='nearest').squeeze(0)

            return image, label, orig_size, crop_size

        else: # infer
            image = sitk.ReadImage(os.path.join(self.folder, self.mode, self.data_name, self.image_names[index]))

            # ------------------------- to array -------------------------
            image = sitk.GetArrayFromImage(image)
            orig_size = image.shape

            # ------------------------- crop -------------------------
            if image.shape[1] == 512:
                image = image[:, top:-bottom, left:-right]
            crop_size = image.shape

            # -------------------- clip and normalize(image only) --------------------
            image = (image - min_bound) / (max_bound - min_bound)
            image[image > 1] = 1
            image[image < 0] = 0

            image = (image * 2) - 1

            # ------------------------- to tensor -------------------------
            image = torch.from_numpy(image).unsqueeze(0).float()

            # ------------------------- downsizing -------------------------
            image = F.interpolate(image.unsqueeze(0), size=self.down_size, mode='trilinear', align_corners=True).squeeze(0)

            return image, orig_size, crop_size

    def __len__(self):
        return len(self.image_names)


def get_loader(data_dir, data_name, data_type, crop_range, hu_range, down_size, num_workers, shuffle, mode, epoch=0, n_splits=0):
    dataset = Dataset(data_dir, data_name, data_type, crop_range, hu_range, down_size, mode, epoch, n_splits)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=1,
                                               num_workers=num_workers,
                                               shuffle=shuffle)

    if mode in ['train', 'val']:
        return dataloader

    else:
        names = dataset.image_names
        return dataloader, names
