import os 
import cv2
import numpy as np 
import shutil 
import sys 
sys.path.insert(0,os.path.realpath('..'))
sys.path.insert(0,os.path.join(os.path.realpath('..'),'piano_utils'))
#from tools.warper import order_points 
from config import cfg 
from piano_utils.networks import PSPNet 
from piano_utils.util import colorize_mask,order_points 
from piano_utils.keyboard import KeyBoard
from PIL import Image 
from tqdm import tqdm 
import shapely
from shapely.geometry import Polygon,MultiPoint
import time 
from skimage.measure import label, regionprops
from collections import Counter
import json 
from IPython import embed
import pickle
from keyboard_exp import HoughKeyBoard 

tmp_dir = '/home/data/lj/Piano/experment/keyboard/tmp_dir'

class HoughTransform(object):
    def __init__(self):
        self.theta_thersh = 0.08

    def hough_transform(self,img):
        res = {}
        img_ori = img.copy()
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150, 3)
        lines = cv2.HoughLines(edges, 1.0, np.pi / 180, 120)
        thetas = [x[0][1] for x in lines if not (x[0][1] < (np.pi / 4.) or
                  x[0][1] > (3.*np.pi/4.0))]
        dic = dict(Counter(thetas))
        theta = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        if len(theta) > 0 and theta[0][1] > 1:   #---统计角度最多重复的直线
            most_theta = theta[0][0]
        else:
            return 
        x_loc, y_loc, pts = [], [], []
        for line in lines:
            rho, theta = line[0]
            if abs(most_theta * 180 / np.pi - 90) > 1.5:  #--键盘是斜着的
                if abs(theta - most_theta) > self.theta_thersh:
                    continue
            else:    #---其他情况
                if not theta == most_theta:
                    continue
            pt1 = (0, max(int(rho / np.sin(theta)), 0))
            pt2 = (img_ori.shape[1], max(int((rho - img_ori.shape[1] * np.cos(theta)) / np.sin(theta)),0))
            cv2.line(img_ori, pt1, pt2, (0, 255, 0), 1)
            pts.append((pt1, pt2))
        return img_ori



class KeyBoard_Exp(KeyBoard):
    def __init__(self):
        KeyBoard.__init__(self)
        print('KeyBoard load finish') 

    def detect(self,img):
        image = img.convert('RGB')
        self.prediction = self.inference(img)
        contours,_ = self.find_contours(image,self.prediction)
        rect = order_points(contours).reshape(-1,1,2).astype(int)
        return rect 

    def mask2image(self,image):
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        w, h = image.size
        colorized_mask = colorize_mask(self.prediction, self.palette)
        #output_im = cv2.cvtColor(np.asarray(colorized_mask),cv2.COLOR_RGB2BGR) 
        #output_im = Image.new('RGB', (w*2, h))
        #output_im.paste(image, (0,0))
        #output_im.paste(colorized_mask, (w,0))
        #output_im = cv2.cvtColor(np.asarray(output_im),cv2.COLOR_RGB2BGR) 
        return colorized_mask 



def draw_line(img,rect):
    if len(rect)!=4:return 
    if len(rect[0])==1:
        rect = [tuple(x[0]) for x in rect]
    rect = sorted(rect,key=lambda x:x[0])
    cv2.line(img,rect[0],rect[1],(0,0,255),2)
    cv2.line(img,rect[2],rect[3],(0,0,255),2)
    if rect[0][1]<rect[1][1]:
        if rect[2][1]<rect[3][1]:
            cv2.line(img,rect[0],rect[2],(0,0,255),2)
            cv2.line(img,rect[1],rect[3],(0,0,255),2)
        else:
            cv2.line(img,rect[0],rect[3],(0,0,255),2)
            cv2.line(img,rect[1],rect[2],(0,0,255),2)
    else:
        if rect[2][1]<rect[3][1]:
            cv2.line(img,rect[1],rect[2],(0,0,255),2)
            cv2.line(img,rect[0],rect[3],(0,0,255),2)
        else:
            cv2.line(img,rect[1],rect[3],(0,0,255),2)
            cv2.line(img,rect[0],rect[2],(0,0,255),2)


def main():
    hour_detector = HoughKeyBoard()
    hourtransform = HoughTransform()
    keyboard_net = KeyBoard_Exp()
    
    img_path = os.path.join(tmp_dir,'tmp.png')
    img = cv2.imread(img_path)
    img_copy = img.copy()

    img_input = Image.fromarray(cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB))
    seg_rect = keyboard_net.detect(img_input)
    for rect in seg_rect:
        rect = rect[0]
        cv2.circle(img_copy,(rect[0],rect[1]),5,(0,0,255),3)
    #draw_line(img_copy,seg_rect)
    img_mask = keyboard_net.mask2image(img_copy)
    img_mask.save(os.path.join(tmp_dir,'mask.png'))
    cv2.imwrite(os.path.join(tmp_dir,'seg_detect.jpg'),img_copy)

    hour_rect = hour_detector.detect_keyboard(img)
    img_line = hourtransform.hough_transform(img)
    for rect in hour_rect:
        cv2.circle(img,(rect[0],rect[1]),5,(0,0,255),3)
    #draw_line(img,hour_rect)
    cv2.imwrite(os.path.join(tmp_dir,'hour_detect.jpg'),img)
    cv2.imwrite(os.path.join(tmp_dir,'hour_line.jpg'),img_line)

if __name__=='__main__':
    main()
