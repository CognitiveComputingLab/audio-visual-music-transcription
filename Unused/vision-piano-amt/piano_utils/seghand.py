import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import sys 
sys.path.insert(0,os.path.abspath(__file__))
import cv2
from networks import PSPNet
from config import cfg 
from util import colorize_mask
import time 
from IPython import embed 

class SegHand(object):
    """docstring for KeyBoard"""
    def __init__(self):
        super(SegHand, self).__init__()
        self.load_handseg_model()
        #print('->>finish seg hand model load')

    def load_handseg_model(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(cfg.MEAN, cfg.STD)
        self.num_classes = cfg.HAND_SEG_NUM_CLASSES
        self.palette = cfg.HAND_SEG_PALETTE

        self.model = PSPNet(num_classes=self.num_classes,backbone='resnet50')
        availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda' if len(availble_gpus) > 0 else 'cpu')

        checkpoint = torch.load(cfg.HAND_SEG_MODEL)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(model)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()


    def segment_detect_hand(self,img,rect):
        resize = False
        with torch.no_grad():
            width,height = img.size 
            cropx2,cropy2 = rect[2],min(height,rect[3]+60)
            cropx1,cropy1 = rect[0],rect[1]
            crop_img = img.crop((cropx1,cropy1,cropx2,cropy2))
            image = crop_img.convert('RGB')
            if resize:
                t1 = time.time()
                iw,ih = image.size
                upsample = nn.Upsample(size=(ih,iw),mode='bilinear',align_corners=True)
                input_img = image.resize((480,ih))
                input = self.normalize(self.to_tensor(input_img)).unsqueeze(0)
                prediction = self.model(input.to(self.device))
                prediction = upsample(prediction.cpu()).squeeze(0)
                prediction = F.softmax(prediction, dim=0).argmax(0).numpy()
                #print('imgsize {} seg hand cost {}'.format(input.size(),time.time()-t1))
            else:
                input = self.normalize(self.to_tensor(image)).unsqueeze(0)
                prediction = self.model(input.to(self.device)).squeeze(0)
                prediction = F.softmax(prediction, dim=0).argmax(0).cpu().numpy()

            self.hand_box, mask = self.post_process(image, prediction, rect)
            self.mask = np.zeros((height,width))
            self.mask[cropy1:cropy2,cropx1:cropx2] = mask 
            return self.hand_box,self.mask
    
    
    def post_process(self,image,mask,rect):
        colorized_mask = colorize_mask(mask, self.palette)
        pmask = np.array(colorized_mask)
        contours, hier = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_box = []
        for cidx,cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            if h>25 and y+rect[1]<rect[3]:
                left_up,right_bottom = (int(x),int(y+rect[1])),(int(x+w),int(y+h+rect[1]))
                hand_box.append((left_up,right_bottom))
        return hand_box,pmask


def detect_hand_by_diffimg(base_img,cur_img,rect,thresh=80):
    base_img_ = base_img.copy()
    cur_img_ = cur_img.copy()
    base_img_ = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
    cur_img_ = cv2.cvtColor(cur_img_,cv2.COLOR_BGR2GRAY)
    dif_img = cv2.absdiff(base_img_,cur_img_)
    dif_img = dif_img[rect[1]:rect[3]+80, rect[0]:rect[2]]
    _, dif_img = cv2.threshold(dif_img, thresh, 255, cv2.THRESH_BINARY_INV)        
    kernel = np.ones((10,10), dtype=np.uint8)
    dif_img = cv2.erode(dif_img, kernel=kernel, iterations=1)
    dif_img = cv2.dilate(dif_img, kernel=kernel, iterations=1)
    _, dif_img = cv2.threshold(dif_img, thresh, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(dif_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_box = []

    for cidx,cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        p1,p2 = (x,y+rect[1]),(x+w,y+h+rect[1])
        if h>cfg.HAND_LENGTH and w>cfg.HAND_LENGTH:
            hand_box.append((p1,p2))
    return hand_box,dif_img

