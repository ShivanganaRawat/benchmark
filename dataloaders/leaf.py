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
from glob import glob
import csv

class LeafDataset(BaseDataSet):
	def __init__(self, create_dataset_using_txt=False, load_from=None, mask_dir=None, **kwargs):
		self.num_classes = 2
		self.palette = [0,0,0,0,128,0]
		self.create_dataset_using_txt = create_dataset_using_txt
		self.load_from = load_from
		self.mask_dir = mask_dir
		super(LeafDataset, self).__init__(**kwargs)


	def _set_files(self):
		self.image_dir = os.path.join(self.root, "images", self.split)
		self.label_dir = os.path.join(self.root, "annotations", self.split)
		self.image_names = []
		self.label_names = []
		if self.create_dataset_using_txt == True:
			file_list = csv.reader(open(self.load_from, 'r'))
			file_names = [r[0] for r in file_list]
			for file in file_names:
				self.image_names.append(file + ".png")
				self.label_names.append(file + "_label.png")
		else:
			self.image_names = sorted([f for f in os.listdir(self.image_dir) if not f.startswith('.')])
			self.label_names = sorted([f for f in os.listdir(self.label_dir) if not f.startswith('.')])
		self.files = list(zip(self.image_names, self.label_names))

	def _load_data(self, index):
		image_name, label_name = self.files[index]
		image_path = os.path.join(self.image_dir, image_name)
		label_path = os.path.join(self.label_dir, label_name)
		image = np.asarray(Image.open(image_path), dtype=np.float32)[:,:,:3]
		label = np.asarray(Image.open(label_path), dtype=np.int32)
		image_id = self.image_names[index].split(".")[0]
		return image, label, image_id


class Leaf(BaseDataLoader):
	def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
					shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, 
					create_dataset_using_txt=False, load_from=None,mask_dir=None):

		self.MEAN = [0.24152601, 0.23049359, 0.17779869]
		self.STD = [0.19978444, 0.18484720, 0.15231922]

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

		self.dataset = LeafDataset(create_dataset_using_txt=create_dataset_using_txt, load_from=load_from, 
				mask_dir=None, **kwargs)
		super(Leaf, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)