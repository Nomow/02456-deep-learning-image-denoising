from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2

class AutoEncoderDataset(Dataset):    
    def __init__(self, img_path, preprocess_transforms = None, aug_transforms = None):
        """  Image dataset class
        Args:
            imgs (np.array(N, h, w, c): images, (N is number of images)
            preprocess_transforms (albumentations.Compose): preprocess transforms
            aug_transforms (albumentations.Compose): augmentation transforms
        """

        self.aug_transforms = aug_transforms
        self.img_file_path = []
        self.preprocess_transforms = preprocess_transforms

        # stores full path of images
        self.img_files = os.listdir(img_path);
        for file in  self.img_files:
            self.img_file_path.append(os.path.join(img_path, file))
            
    def __len__(self):
        return len(self.img_file_path)

    def __getitem__(self, idx):
        img_in = cv2.imread(self.img_file_path[idx], cv2.COLOR_BGR2RGB)
        img_out = img_in.copy()
        # applies transformations
        if self.aug_transforms:
            img_out = self.aug_transforms(image=img_out)['image']
            
        if self.preprocess_transforms:
            img_out = self.preprocess_transforms(image=img_out)['image']
            img_in = self.preprocess_transforms(image=img_in)['image']

            
        return img_out, img_in
    
    
class Cifar10AutoEncoderDataset(Dataset):    
    def __init__(self, imgs, preprocess_transforms = None, aug_transforms = None):
        """  Cifar10 Dataset class
        Args:
            imgs (np.array(N, h, w, c): images, (N is number of images)
            preprocess_transforms (albumentations.Compose): preprocess transforms
            aug_transforms (albumentations.Compose): augmentation transforms
        """

        self.aug_transforms = aug_transforms
        self.preprocess_transforms = preprocess_transforms
        self.imgs = imgs
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_in = self.imgs[idx]
        img_out = img_in.copy()
        # applies transformations
        if self.aug_transforms:
            img_out = self.aug_transforms(image=img_out)['image']
            
        if self.preprocess_transforms:
            img_out = self.preprocess_transforms(image=img_out)['image']
            img_in = self.preprocess_transforms(image=img_in)['image']
        return img_out, img_in
    
    
