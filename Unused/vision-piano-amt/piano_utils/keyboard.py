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
import cv2 
sys.path.insert(0,os.path.abspath(__file__))

from networks import PSPNet
from config import cfg 
from util import colorize_mask,calAngle,order_points 
from IPython import embed 
    
class KeyBoard(object):
    """docstring for KeyBoard"""
    def __init__(self):
        super(KeyBoard, self).__init__()
        self.load_keyboard_model()
        #print('->>finish keyborad model load')
        
        self.scales = [0.5,0.75,1.0]

    def load_keyboard_model(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(cfg.MEAN, cfg.STD)
        self.num_classes = cfg.KEYBOARD_NUM_CLASSES
        self.palette = cfg.KEYBOARD_PALETTE
        
        self.model = PSPNet(num_classes=self.num_classes)
        availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        checkpoint = torch.load(cfg.KEYBOARD_MODEL)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'module' in list(checkpoint.keys())[0] and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(model)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    
    def multi_scale_predict(self,image):
        input_size = (image.size(2),image.size(3))
        upsample = nn.Upsample(size=input_size,mode='bilinear',align_corners=True)

        total_predictions = np.zeros((self.num_classes, image.size(2), image.size(3)))

        image = image.data.data.cpu().numpy()
        for scale in self.scales:
            scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
            scaled_img = torch.from_numpy(scaled_img).to(self.device)
            scaled_prediction = upsample(self.model(scaled_img).cpu())
            total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)
        
        total_predictions /= len(self.scales)
        return total_predictions

    def detect_keyboard(self,img):
        image = img.convert('RGB') 
        prediction = self.inference(img)
        result = self.post_process(image,prediction)
        if not result['flag']:return result
        if result['keyboard_rect'] is not None: return result 
        rotated_img = result['rotated_img']
        img = Image.fromarray(cv2.cvtColor(rotated_img,cv2.COLOR_BGR2RGB))
        prediction = self.inference(img)

        warp_result = self.post_process1(img,prediction)
        warp_result['rotated_img'] = rotated_img 
        warp_result['rote_M'] = result['rote_M']
        if not warp_result['flag']:return warp_result
        if warp_result['keyboard_rect'] is not None: return warp_result
        warp_img = warp_result['warp_img']
        
        img = Image.fromarray(cv2.cvtColor(warp_img,cv2.COLOR_BGR2RGB))
        prediction = self.inference(img)
        fin_result = self.post_process2(img,prediction)
        if not fin_result['flag']:return fin_result
        fin_result['warp_M'] = warp_result['warp_M']
        fin_result['rote_M'] = result['rote_M']
        fin_result['warp_img'] = warp_img 
        fin_result['rotated_img'] = rotated_img
        return fin_result 
    
    def inference(self,img):
        with torch.no_grad():
            image = img.convert('RGB')
            input = self.normalize(self.to_tensor(image)).unsqueeze(0)
            prediction = self.multi_scale_predict(input)
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
        return prediction 
        
    def find_rect(self,pmask,sx,sy,ex,ey):
        height,width = pmask.shape
        loc_x,loc_y = [],[]
        for i in range(sy,ey):
            for j in range(sx,ex):
                if pmask[i,j]!=0:
                    loc_y.append(i)
        loc_y.sort()
        loc_y = np.unique(np.array(loc_y))
        locy_min,locy_max = 0,0
        for y in loc_y:
            cmask = np.where(pmask[y]!=0)[0]
            if len(cmask)>0.3*width:
                locy_min = y 
                break 
        for y in loc_y[::-1]:
            cmask = np.where(pmask[y]!=0)[0]
            if len(cmask)>0.3*width:
                locy_max = y 
                break 
        piano_ylen = locy_max-locy_min 
        locx_min,locx_max = 0,0
        for x in range(sx,ex):
            cmask = np.where(pmask[locy_min:locy_max,x]!=0)[0]
            if len(cmask)>0.3*(piano_ylen):
                locx_min = x 
                break 
        for x in range(sx,ex)[::-1]:
            cmask = np.where(pmask[locy_min:locy_max,x]!=0)[0]
            if len(cmask)>0.3*piano_ylen:
                locx_max = x 
                break
        Rect = (locx_min,locy_min,locx_max,locy_max)
        if locy_max-locy_min<20:
            return False,Rect 
        return True,Rect 
    

    def find_contours(self,image,mask):
        w, h = image.size 
        colorized_mask = colorize_mask(mask, self.palette)
        #colorized_mask.save('mask.png')
        pmask = np.array(colorized_mask)
        pmask[pmask==1] = 255 
        _,base_img = cv2.threshold(pmask,150,255,cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(base_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        board_contours = np.array([])
        for contour in contours:
            contour = np.array(contour,dtype=np.int32)
            if len(contour)>len(board_contours):
                board_contours = contour
        contours = np.squeeze(board_contours)
        return contours,pmask 

    def post_process(self,image,mask):
        w, h = image.size 
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        contours,pmask = self.find_contours(image,mask)
        result = {}
        rect = order_points(contours)
        if len(contours)>500:
            lt,rt,rb,lb = rect            
            if abs(lt[1]-rt[1])>5 or abs(rb[1]-lb[1])>5:
                xb1,yb1,xb2,yb2 = lb[0],lb[1],rb[0],rb[1]
                xt1,yt1,xt2,yt2 = lt[0],lt[1],rt[0],rt[1]
                center = (w//2,h//2)
                if abs(yb1-yb2)>abs(yt1-yt2):
                    angle = calAngle(xb1,yb1,xb2,yb2)
                    M = cv2.getRotationMatrix2D(center,angle,1)
                    rotated_img = cv2.warpAffine(img,M,(w,h))
                else:
                    angle = calAngle(xt1,yt1,xt2,yt2)
                    M = cv2.getRotationMatrix2D(center,angle,1)
                    rotated_img = cv2.warpAffine(img,M,(w,h))
                result = {'flag':1,'rote_M':M,'warp_M':None,'keyboard_rect':None,
                        'rotated_img':rotated_img 
                }
            else:
                lr,rt,rb,lb = rect
                sx,ex = int(min(lt[0],lb[0])),int(max(rt[0],rb[0]))
                sy,ey = int(min(lt[1],rt[1])),int(max(lb[1],rb[1]))
                flag,keyboard_rect = self.find_rect(pmask,sx,sy,ex,ey)
                result = {
                        'flag':flag,
                        'rote_M':None,
                        'warp_M':None,
                        'keyboard_rect':keyboard_rect,
                        'rotated_img':None 
                }

        else:
            result = {'flag':0,
                    'rote_M':None,
                    'warp_M':None,
                    'keyboard_rect':None,
                    'rotated_img':None}
        return result 

    def post_process1(self,image,mask):
        w, h = image.size 
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        contours,pmask = self.find_contours(image,mask)
        result = {}
        rect = order_points(contours)
        if len(contours)>500:
            lt,rt,rb,lb = rect
            if abs(lt[1]-rt[1])>5 or abs(rb[1]-lb[1])>5:
                xb1,yb1,xb2,yb2 = lb[0],lb[1],rb[0],rb[1]
                xt1,yt1,xt2,yt2 = lt[0],lt[1],rt[0],rt[1]
                if abs(yb1-yb2)>abs(yt1-yt2):
                    pts1 = np.float32([lt,lb,rt,rb])
                    if yb1>yb2:
                        pts2 = np.float32([lt,lb,rt,[rb[0],lb[1]]])
                    else:
                        pts2 = np.float32([lt,[lb[0],rb[1]],rt,rb])
                    M=cv2.getPerspectiveTransform(pts1,pts2)
                    warp_img=cv2.warpPerspective(img,M,(w,h))
                else:
                    pts1 = np.float32([lt,lb,rt,rb])
                    if yt1<yt2:
                        pts2 = np.float32([lt,lb,[rt[0],lt[1]],rb])
                    else:
                        pts2 = np.float32([[lt[0],rt[1]],lb,rt,rb])
                    M=cv2.getPerspectiveTransform(pts1,pts2)
                    warp_img=cv2.warpPerspective(img,M,(w,h))
                result = {
                        'flag':1,
                        'warp_M':M,
                        'keyboard_rect':None,
                        'warp_img':warp_img
                }
            else:
                lr,rt,rb,lb = rect
                sx,ex = int(min(lt[0],lb[0])),int(max(rt[0],rb[0]))
                sy,ey = int(min(lt[1],rt[1])),int(max(lb[1],rb[1]))
                flag,keyboard_rect = self.find_rect(pmask,sx,sy,ex,ey)
                result = {
                        'flag':flag,
                        'warp_M':None,
                        'keyboard_rect':keyboard_rect,
                        'warp_img':None 
                }

        else:
            result = {'flag':0,'warp_M':None,'keyboard_rect':None,'warp_img':None}
        return result 

    def post_process2(self,image,mask):
        w, h = image.size 
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        contours,pmask = self.find_contours(image,mask)
        result = {}
        rect = order_points(contours)
        if len(contours)>500:
            lt,rt,rb,lb = rect
            sx,ex = int(min(lt[0],lb[0])),int(max(rt[0],rb[0]))
            sy,ey = int(min(lt[1],rt[1])),int(max(lb[1],rb[1]))
            flag,keyboard_rect = self.find_rect(pmask,sx,sy,ex,ey)
            result = {'flag':flag,'keyboard_rect':keyboard_rect}
        else:
            result = {'flag':0,'keyboard_rect':None}
        return result 

        
