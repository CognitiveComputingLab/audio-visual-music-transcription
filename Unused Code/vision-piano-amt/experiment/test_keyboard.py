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

exp_cfg = {
        'exp_imgs':'/home/data/lj/Piano/experment/keyboard/exp_imgs',
        'tmp_dir':'/home/data/lj/Piano/experment/keyboard/tmp_dir',
        'res_dir':'/home/data/lj/Piano/experment/keyboard/res'
}

class HoughKeyBoard(object):
    def __init__(self,binary_thresh=190):
        self.binary_thresh = binary_thresh
        self.theta_thersh = 0.08
        
    def detect_keyboard(self, img, binary_thresh=190):
        img_ori = img.copy()
        hough_result = self.hough_transform(img)
        if not hough_result['flag']:
            result = {'flag': None, 'rote_M': None, 'warp_M': None, 'keyboard_rect': None, 'rotated_img': None}
            return None
        crop_img = hough_result['crop_img']
        if crop_img is None:return None 
        h, w, _ = crop_img.shape
        if h==0 or w==0: return None 
        #---find connected domain
        label_img = self.binary_img(crop_img,thresh=binary_thresh)
        result = self.post_process(crop_img, label_img)
        if not result['flag']:return None
        if hough_result['y_loc'] < 0:
            offset = img_ori.shape[0] + hough_result['y_loc']
        else:
            offset = hough_result['y_loc']

        rect_point = []
        if result['keyboard_rect'] is not None:   #--不需旋转
            rect = result['keyboard_rect']
            lt, rt, rb, lb = (rect[0], rect[1]), (rect[2], rect[1]), (rect[0], rect[3]), (rect[2], rect[3])
            points = (lt, rt, rb, lb)
            for point in points:
                rect_point.append((int(point[0]), int(point[1] + offset)))
            return rect_point

        #---需要旋转---
        for point in result['rect']:
            rect_point.append((int(point[0]), int(point[1] + offset)))
        return rect_point 

    def binary_img(self, img, thresh):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('./gray.jpg', gray)
        if not thresh == self.binary_thresh:
            kernel = np.ones((5, 5), np.uint8)
            gary = cv2.dilate(gray, kernel, 2)
        label_img = label(gray, neighbors=4, background=0)
        return label_img

    def hough_transform(self,img):
        #---霍夫变化
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
            res = {'flag': False, 'crop_img': None, 'y_loc': None}
            return res
            
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

        # cv2.imwrite('./test_imgs.jpg', img_ori)
        crop_img, y_loc = self.find_keyboardloc(img, pts, most_theta)
        # cv2.imwrite('./crop_img.jpg', crop_img)
        res = {'flag': True, 'crop_img': crop_img, 'y_loc': y_loc}
        return res

    def find_keyboardloc(self, img, pts, most_theta):
        y1min = min([x[0][1] for x in pts])
        y1max = max([x[0][1] for x in pts])
        y2min = min([x[1][1] for x in pts])
        y2max = max([x[1][1] for x in pts])
        if most_theta * 180 / np.pi > 90:
            crop_img = img[y1min:y2max,:]
            return crop_img, y1min
        else:
            crop_img = img[y2min:y1max,:]  #---img[-977:1790]  符号相当于反着取,就是从1790倒着取977的长度，977就是h
            return crop_img,y2min

    def post_process(self, img, pmask):
        h, w = img.shape[:2]
        # pmask = cv2.cvtColor(pmask.astype(np.uint8), cv2.COLOR_BGR2GRAY)  #---需转换为np.uint8类型
        props = regionprops(pmask)
        pixel_nums = np.array([int(prop.area) for prop in props])
        ind = np.argsort(-pixel_nums)  #--降序排列
        final_indx = np.argsort(ind)  #对降序排列后的inx升序排列,相当于对原数组中的位置进行了排列，现在ind对应着原数组中元素的排名,0表示最大
        
        first_idx = 0
        if not len(final_indx) > 0:
            result = {'flag': 0, 'keyboard_rect': None, 'rotated_img': None}
            return result
        index = int(np.where(final_indx == first_idx)[0])
        pmask[pmask != (index + 1)] = 0
        pts = np.array([np.array([x[1], x[0]]) for x in list(props[index].coords)])
        rect = self.order_points(pts)

        while rect[0][1] < 0.02* h:  #---因为二值化后有些图像上面也会有一些白色像素(面积也比较大)
            first_idx += 1
            tmp = np.where(final_indx == first_idx)[0]
            if len(tmp)==0:
                return {'flag': 0, 'keyboard_rect': None, 'rotated_img': None}
            index = int(np.where(final_indx == first_idx)[0])
            pmask[pmask != (index + 1)] = 0
            pts = np.array([np.array([x[1], x[0]]) for x in list(props[index].coords)])
            rect = self.order_points(pts)
        
        #---test rect
        img_draw = img.copy()
        ymin, xmin, ymax, xmax = props[index].bbox
        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        # cv2.imwrite('./rect.jpg', img_draw)

        result = {}
        if len(pts) > 10000:
            lt, rt, rb, lb = rect
            if abs(lt[1] - rt[1]) > 5 or abs(lb[1] - rb[1]) > 5:
                result = {'flag': 1, 'keyboard_rect': None, 'rect': rect}
            else:
                lr,rt,rb,lb = rect
                sx, ex = int(min(lt[0], lb[0])), int(max(rt[0], rb[0]))
                sy, ey = int(min(lt[1], rt[1])), int(max(lb[1], rb[1]))
                flag,keyboard_rect = self.find_box(pmask,sx,sy,ex,ey)                    
                result = {
                        'flag':flag,
                        'keyboard_rect':keyboard_rect,
                        'rect':rect
                }
        else:
            result = {'flag': 0, 'rect': rect}
        return result

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32) #--lt rt rb lb
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def calAngle(self, x1, y1, x2, y2): 
        import math 
        angle = 0.0
        dy = y1 - y2
        dx = x1 - x2
        angle = math.atan(dy / dx)
        return (angle * 180 / math.pi)
        
    def find_box(self, pmask, sx, sy, ex, ey):
        height,width = pmask.shape
        loc_x, loc_y = [], []
        for i in range(sy,ey):
            for j in range(sx,ex):
                if pmask[i,j]!=0:
                    loc_y.append(i)   
        loc_y.sort()
        loc_y = np.unique(np.array(loc_y))
        locy_min, locy_max = 0, 0
        #---仅仅适用于刚开始出现手的情况
        width_thresh=0.3  #---对于钢琴键盘通过二值化检测而言的话,由于黑键那部分是黑像素,因此每行的像素之和阈值可以调少一点
        for y in loc_y:   
            cmask = np.where(pmask[y] != 0)[0]
            if len(cmask)>width_thresh*width:   
                locy_min = y     #---从头开始遍历,得到键盘的起始y_loc
                break 
        for y in loc_y[::-1]:   #----倒序开始遍历,得到键盘的终止位置
            cmask = np.where(pmask[y]!=0)[0]
            if len(cmask)>width_thresh*width:
                locy_max = y 
                break
        piano_ylen = locy_max - locy_min
        
        locx_min, locx_max = 0, 0
        #---刚好键盘左右两边都是白键,因此判断键盘列的坐标不存在和上面一样的问题
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
        Rect = (locx_min, locy_min, locx_max, locy_max)
        if locy_max-locy_min<20:    
            return False,Rect 
        return True,Rect 

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


def get_exp_imgs(path='/home/data/lj/Piano/paperData/IWSSIP/TrainSet/images'):
    subpaths = [os.path.join(path,x) for x in os.listdir(path)]
    for subpath in subpaths:
        img_files = [os.path.join(subpath,x) for x in os.listdir(subpath)]
        img_files.sort()
        img_dir = os.path.split(subpath)[-1].split('_img')[0]  #--os.path.split()分离路径和文件名        
        dst_path = os.path.join(exp_cfg['exp_imgs'],img_dir+'.jpg')
        shutil.copyfile(img_files[0],dst_path)

class KeyBoard_Exp(KeyBoard):
    def __init__(self):
        KeyBoard.__init__(self)
        print('KeyBoard load finish') 

    def detect(self,img):
        image = img.convert('RGB')
        prediction = self.inference(img)
        contours,_ = self.find_contours(image,prediction)
        rect = order_points(contours).reshape(-1,1,2).astype(int)
        return rect 

def cal_iou(gt_rect,det_rect):
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



def main():
    path = exp_cfg['exp_imgs']
    img_files = [os.path.join(path,x) for x in os.listdir(path)]
    gt_box_dict = get_img_box_dict()
    
    keyboard_net = KeyBoard_Exp()
    hourdetect = HoughKeyBoard()
    seg_box_dict = dict()
    hour_box_dict = dict()

    error_list = []
    for img_path in tqdm(img_files):
        img_mark = os.path.basename(img_path).split('.')[0]
        # if not img_mark == 'level_1_no_1':
        #     continue
        if img_mark in gt_box_dict.keys():
            img = Image.open(img_path)
            det_rect = keyboard_net.detect(img)
            seg_box_dict[img_mark] = det_rect 
            
            opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) 
            hour_rect = hourdetect.detect_keyboard(opencv_img)
            if hour_rect is None:
                try:
                    hour_rect = hourdetect.detect_keyboard(opencv_img, binary_thresh=90)
                    if hour_rect is not None:
                        hour_rect = np.array(hour_rect).reshape(-1, 1, 2).astype(int)
                except Exception as e:   #--exception可以包括所有的异常类型  
                    pass   #---不管异常
            else:
                hour_rect = np.array(hour_rect).reshape(-1, 1, 2).astype(int)
            
            if not hour_rect is None: 
                hour_box_dict[img_mark] = hour_rect
            
            if not hour_rect is None:
                hour_rect = np.array(hour_rect).reshape(4, 2).astype(int)
                for point in hour_rect:
                    cv2.circle(opencv_img, (point[0],point[1]), 5, (0, 0, 255), 3)
                #file_path = './res/res_{}'.format(os.path.basename(img_path))
                file_path = './{}/res_{}'.format(exp_cfg['res_dir'],os.path.basename(img_path))
                # cv2.imwrite(file_path, opencv_img)
            else:
                error_list.append(img_path)


    for error_path in error_list:
        print(error_path)
    # embed()

    seg_ious = []
    for img_mark,det_rect in seg_box_dict.items():
        gt_rect = gt_box_dict[img_mark]
        iou = cal_iou(gt_rect,det_rect)
        seg_ious.append(iou)

    hour_ious = []
    for img_mark,det_rect in hour_box_dict.items():
        gt_rect = gt_box_dict[img_mark]
        iou = cal_iou(gt_rect,det_rect)
        hour_ious.append(iou)

    print('detect:{}/{}={} segment keyboard mean iou {:.3}'.format(len(seg_box_dict),len(gt_box_dict),
                                                        len(seg_box_dict)/len(gt_box_dict),np.mean(seg_ious)))
    print('detect:{}/{}={} hourghline keyboard mean iou {:.3}'.format(len(hour_box_dict),len(gt_box_dict),
                                                        len(hour_box_dict)/len(gt_box_dict),np.mean(hour_ious)))

if __name__=='__main__':
    main()
