from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2

class AutoEncoderDataset(Dataset):    
    def __init__(self, img_path, transform = None):
        """  Full court dataset class for training smp models
        Args:
            img_path (str): path to the image folder
            transform (albumentations.Compose): augmentation transforms to apply to images
        """

        self.transform = transform
        self.img_file_path = []

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
        if self.transform:
            sample = self.transform(image=img_out)
            img_out = sample['image']
            
        return img_in, img_out
    
    
class Cifar10AutoEncoderDataset(Dataset):    
    def __init__(self, imgs, transform = None):
        """  Full court dataset class for training smp models
        Args:
            imgs (np.array(N, h, w, c): images, (N is number of images)
            transform (albumentations.Compose): augmentation transforms to apply to images
        """

        self.transform = transform
        self.imgs = imgs
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_in = self.imgs[idx]
        img_out = img_in.copy()
        # applies transformations
        if self.transform:
            sample = self.transform(image=img_out)
            img_out = sample['image']
            
        return img_in, img_out