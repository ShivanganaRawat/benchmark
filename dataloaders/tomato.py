# Originally written by Kazuto Nakashima 
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import csv


class TomatoDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, create_dataset_using_txt=False, load_from=None, **kwargs):
        self.num_classes = 2
        self.palette = [0, 0, 0, 255, 0, 0]
        self.create_dataset_using_txt = create_dataset_using_txt
        self.load_from = load_from
        super(TomatoDataset, self).__init__(**kwargs)

    def _set_files(self):

        image_dir = os.path.join(self.root, "images")
        label_dir = os.path.join(self.root, "labels")

        if self.create_dataset_using_txt:
            file_list_path = self.load_from
        else:
            file_list_path = os.path.join(self.root, self.split + ".csv")
        file_set = csv.reader(open(file_list_path, 'rt'))
        file_list = [r[0] for r in file_set]
        
        self.image_path, self.label_path = [], []
        for file in file_list:
            self.image_path.append(os.path.join(image_dir, self.split, file+".jpg"))
            self.label_path.append(os.path.join(label_dir, self.split, file+"_label.png"))
        self.image_path.sort()
        self.label_path.sort()
        self.files = list(zip(self.image_path, self.label_path))
    
    def _load_data(self, index):
        image_path = self.files[index][0]
        label_path = self.files[index][1]
        image = np.asarray(Image.open(image_path), dtype=np.uint8)
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        if label.shape[0] != image.shape[0]:
            image = np.moveaxis(image,0,1)[:, ::-1, :]
        image_id = self.files[index][0].split("/")[-1].split(".")[0]
        if label.shape[0]>label.shape[1]:
            image = cv2.resize(image, (int((label.shape[1]/label.shape[0])*512), 512), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (int((label.shape[1]/label.shape[0])*512), 512), interpolation=cv2.INTER_NEAREST)
        else:
            image = cv2.resize(image, (512, int((label.shape[0]/label.shape[1])*512)), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (512, int((label.shape[0]/label.shape[1])*512)), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32)
        label = label.astype(np.int32)
        #print(self.split,image_path)
        #print(self.split,label.shape)
        return image, label, image_id


class Tomato(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, 
                    create_dataset_using_txt=False, load_from=None):
        
        self.MEAN = [0.46723480, 0.49984341, 0.31839238]
        self.STD = [0.22013583, 0.19797440, 0.24449959]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        self.dataset = TomatoDataset(create_dataset_using_txt=create_dataset_using_txt, load_from=load_from, **kwargs)
        super(Tomato, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

        
        
        
        
        
        
        
        


'''
class TomatoDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, **kwargs):
        self.num_classes = 21
        self.palette = [0, 0, 0, 255, 0, 0]
        super(TomatoDataset, self).__init__(**kwargs)

    def _set_files(self):

        image_dir = os.path.join(self.root, "images")
        label_dir = os.path.join(self.root, "labels")

        file_list_path = os.path.join(self.root, self.split + ".csv")
        file_set = csv.reader(open(file_list_path, 'rt'))
        file_list = [r[0] for r in file_set]
        
        self.image_path, self.label_path = [], []
        for file in file_list:
            self.image_path.append(os.path.join(image_dir, file+".jpg"))
            self.label_path.append(os.path.join(label_dir, file+"_label.jpg"))
        self.image_path.sort()
        self.label_path.sort()
        self.files = list(zip(self.image_path, self.label_path))
    
    def _load_data(self, index):
        image_path = self.files[index][0]
        label_path = self.files[index][1]
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index][0].split("/")[-1].split(".")[0]
        return image, label, image_id


class Tomato(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        self.dataset = TomatoDataset(**kwargs)
        super(Tomato, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
'''
