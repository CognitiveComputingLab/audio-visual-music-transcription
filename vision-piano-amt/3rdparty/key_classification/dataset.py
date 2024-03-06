import os 
from PIL import Image 
import torch 
import torch.nn as nn 
import torch.utils.data as data
import numpy as np 
import cv2 

class KeyPressDataset(data.Dataset):
    def __init__(self, file_name, transform=None):
        super(KeyPressDataset, self).__init__()
        self.file_name = file_name 

        self.fnames = []
        self.labels = []
        if self.file_name.endswith('txt'):
            with open(file_name,'r') as f:
                for item in f.readlines():
                    item = item.strip().split(' ')
                    if len(item)>0:
                        self.fnames.append(item[0])
                        self.labels.append(int(item[1]))
        elif self.file_name.endswith('npz'):
            data = np.load(self.file_name)
            self.imgs = data['imgs']
            self.labels = data['labels']

        self.transform = transform 
    
    
    def __len__(self):
        return len(self.labels) 

    def __getitem__(self,index):
        if self.file_name.endswith('txt'):
            img_name = self.fnames[index]
            label = self.labels[index]

            img = Image.open(img_name)
            if self.transform is not None:
                img = self.transform(img)

            label = torch.LongTensor([label])
            return img,label
        elif self.file_name.endswith('npz'):
            img = self.imgs[index].astype(np.uint8)
            label = self.labels[index]
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            if self.transform is not None:
                img = self.transform(img)
            label = torch.LongTensor([label])
            return img,label 
        
