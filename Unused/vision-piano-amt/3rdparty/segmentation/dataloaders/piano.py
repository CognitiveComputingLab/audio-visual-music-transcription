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

ignore_label = 255 
ID_TO_TRAINID = {2:ignore_label,0:0,1:1}

class PianoDataset(BaseDataSet):
    def __init__(self,**kwargs):
        self.num_classes = 2 
        self.palette = palette.piano_palette 
        self.id_to_trainId = ID_TO_TRAINID 
        super(PianoDataset,self).__init__(**kwargs)

    def _set_files(self):
        file_list = os.path.join(self.root,self.split+'.txt')
        with open(file_list,'r') as f:
            items = [line.strip() for line in f.readlines()]
        self.img_files = []
        self.label_files = []
        for item in items:
            split_line = item.split('\t')
            self.img_files.append(split_line[0])
            self.label_files.append(split_line[1])
        self.files = self.img_files
    def _load_data(self,index):
        image_path = self.img_files[index]
        label_path = self.label_files[index]
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path),dtype=np.int32)
        image_id = '0'
        for k,v in self.id_to_trainId.items():
            label[label==k] = v
        return image,label,image_id 



class PIANO(BaseDataLoader):
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
    
        self.dataset = PianoDataset(**kwargs)
        super(PIANO, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

