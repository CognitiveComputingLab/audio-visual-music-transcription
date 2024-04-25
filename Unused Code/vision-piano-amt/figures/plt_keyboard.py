import os 
import cv2
import numpy as np 
import shutil 
import sys 
sys.path.insert(0,os.path.realpath('..'))
sys.path.insert(0,os.path.join(os.path.realpath('..'),'piano_utils'))
from tools.warper import order_points 
from config import cfg 
from piano_utils.networks import PSPNet 
from piano_utils.util import colorize_mask 
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

exp_cfg = {
        'exp_imgs':'/home/data/lj/Piano/experment/keyboard/exp_imgs',
        'tmp_dir':'/home/data/lj/Piano/experment/keyboard/tmp_dir',
        'figure_dir':'/home/data/lj/Piano/experment/keyboard_figure'
}
class HoughKeyBoard(object):
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

def get_img_box_dict():
    img_box_dict = dict()
    file_name = '/home/data/lj/Piano/Segment/train.txt'
    with open(file_name,'r') as fr:
        items = [l.strip() for l in fr.readlines()]
    mask_lists = []
    for item in items:
        item = item.split()
        if 'tools' in item[0]:
            #mask_dir = item[0].split('/')[-2]
            continue
        else:
            mask_dir = os.path.basename(item[0]).split('_img_') [0]
        if 'segment' in item[0]:continue 
        if mask_dir in mask_lists:continue
        mask_lists.append(mask_dir)
        img_mask = cv2.imread(item[1],cv2.IMREAD_GRAYSCALE)
        img_mask[img_mask==2] = 1
        img_mask[img_mask==1] = 255
        contours,_ = cv2.findContours(img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        assert len(contours)==1,'value wrong'
        contours = np.squeeze(contours)
        rect = order_points(contours).reshape(-1,1,2).astype(int)
        img_box_dict[mask_dir] = rect
    
    json_path = os.path.join(exp_cfg['exp_imgs'],'need_labels')
    json_files = [os.path.join(json_path,x) for x in os.listdir(json_path) if x.endswith('json')]
    json_files.sort()
    for json_file in json_files:
        with open(json_file,'r') as fr:
            items = json.load(fr)
        basename = os.path.basename(json_file).split('.')[0]
        points = np.array(items['shapes'][0]['points'])
        rect = order_points(points).reshape(-1,1,2).astype(int)
        img_box_dict[basename] = rect 
    return img_box_dict 


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
        output_im = Image.new('RGB', (w*2, h))
        output_im.paste(image, (0,0))
        output_im.paste(colorized_mask, (w,0))
        output_im = cv2.cvtColor(np.asarray(output_im),cv2.COLOR_RGB2BGR) 
        return output_im


def cal_iou(gt_rect, det_rect):
    #---不规则的两个四边形计算Iou,不是矩形了
    gt_rect = gt_rect.reshape(4, 2)
    poly1 = Polygon(gt_rect).convex_hull
    det_rect = det_rect.reshape(4,2)
    poly2 = Polygon(det_rect).convex_hull
    union_poly = np.concatenate((gt_rect,det_rect))
    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area 
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou= 0
            iou=float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou 


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    seg_pickle_file = os.path.join(exp_cfg['tmp_dir'],'seg.pkl')
    hour_pickle_file = os.path.join(exp_cfg['tmp_dir'],'hourgh.pkl')
    path = exp_cfg['exp_imgs']
    save_seg_dir = os.path.join(exp_cfg['figure_dir'],'segment')
    save_hourgh_dir = os.path.join(exp_cfg['figure_dir'],'hourgh')
    ensure_dir(save_seg_dir)
    ensure_dir(save_hourgh_dir)

    img_files = [os.path.join(path,x) for x in os.listdir(path)]
    gt_box_dict = get_img_box_dict()

    with open(seg_pickle_file,'rb') as f1:
        seg_box_dict = pickle.load(f1)
    with open(hour_pickle_file,'rb') as f2:
        hour_box_dict = pickle.load(f2)

    seg_ious = []
    for img_mark,det_rect in seg_box_dict.items():
        gt_rect = gt_box_dict[img_mark]
        iou = cal_iou(gt_rect,det_rect)
        if iou>0.5:
            seg_ious.append(iou)
        else:print(img_mark)

    hour_detector = HoughKeyBoard()
    keyboard_net = KeyBoard_Exp()

    hour_ious = []
    for img_mark,det_rect in hour_box_dict.items():
        gt_rect = gt_box_dict[img_mark]
        iou = cal_iou(gt_rect,det_rect)
        if iou>0.5:
            hour_ious.append(iou)
        else:
            img = cv2.imread(os.path.join(path,img_mark+'.jpg'))
            img_copy = img.copy()
            img_input = Image.fromarray(cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB))
            seg_rect = keyboard_net.detect(img_input)
            for rect in det_rect:
                rect = rect[0]
                cv2.circle(img,(rect[0],rect[1]),5,(0,255,0),3)
            for rect in seg_rect:
                rect = rect[0]
                cv2.circle(img_copy,(rect[0],rect[1]),5,(0,255,0),3)
            img_copy = keyboard_net.mask2image(img_copy)
            img = hour_detector.hough_transform(img)
            cv2.imwrite(os.path.join(save_hourgh_dir,img_mark+'.jpg'),img)
            cv2.imwrite(os.path.join(save_seg_dir,img_mark+'.jpg'),img_copy)

if __name__=='__main__':
    main()
