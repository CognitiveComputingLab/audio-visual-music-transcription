# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np 
from IPython import embed 
from util import find_connect_domain

def remove_region(img):
    if len(img.shape) == 3:
        print("please input a gray image")
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if (i < 0.08 * h or i > (2.0/3) * h):
                img[i, j] = 255
    for i in range(h):
        for j in range(w):
            if (j < 0.005 * w or j > 0.994 * w):
                img[i, j] = 255
    return img

def near_white(white_loc,black_boxes):
    diffs = []
    for i in range(len(black_boxes)):
        diff = abs(black_boxes[i][0] - white_loc)
        diffs.append(diff)
    index = diffs.index(min(diffs))
    return index

def contrast_img(img, c, b):
    rows, cols, channels = img.shape 
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return  dst 

class BwLabel(object):
    def __init__(self):
        super(BwLabel, self).__init__()

    def key_loc(self, base_img):
        white_loc = []
        black_boxes = []
        total_top = []
        total_bottom = [] 
        black_loc = []
        ''' 
        ori_img = base_img.copy()
        height,width,_ = base_img.shape 
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        base_img = remove_region(base_img)
        _, base_img = cv2.threshold(base_img, 150, 255, cv2.THRESH_BINARY) 
        base_img = cv2.GaussianBlur(base_img, (5, 5), 0)
        black_boxes = find_connect_domain(base_img)
        black_loc = [box[0] for box in black_boxes]
        '''
        ori_img = base_img.copy()
        draw_img = base_img.copy()
        height,width,_ = ori_img.shape 
        black_boxes,black_loc = self.find_black_boxes(ori_img)
        if len(black_boxes)!=36:
            ori_img = contrast_img(ori_img,1.3,3)
            black_boxes,black_loc = self.find_black_boxes(ori_img)
        if len(black_boxes)==37:
            area1 = black_boxes[0][2]*black_boxes[0][3]
            area2 = black_boxes[-1][2]*black_boxes[-1][3]
            if area1>area2:
                del black_boxes[-1]
            else:
                del black_boxes[0]
        assert len(black_boxes)==36,'black number is wrong'
        # #----得到白键的区域
        white_loc = self.find_white_loc_old(black_loc,black_boxes,width)
        #print("the number of whitekey_num is {}".format(len(white_loc)))
        #--------找到白键所在的box---
        for i in range(1, len(white_loc)):
            white_x = white_loc[i - 1]
            white_width = white_loc[i] - white_x
            # print(white_x,white_width)
            if i == 1:
                top_box = (white_x, 0, black_boxes[i - 1][0] - white_x, 1.1 * black_boxes[i - 1][3]) #---(x,y,w,h)
                bottom_box=(white_x,1.1*black_boxes[i-1][3],white_width,height-1.1*black_boxes[i-1][3])
                total_top.append(top_box)
                total_bottom.append(bottom_box)
            elif i==2: 
                top_box = (black_boxes[i - 2][0]+black_boxes[i - 2][2], 0, white_loc[i] - (black_boxes[i - 2][0]+black_boxes[i - 2][2]), 1.1 * black_boxes[i - 2][3])
                bottom_box=(white_x,1.1*black_boxes[i-2][3],white_width,height-1.1*black_boxes[i-2][3])
                total_top.append(top_box)
                total_bottom.append(bottom_box)
            elif (i == 3 or ((i - 3) % 7 == 0) and i < 52) or (i == 6 or ((i - 6) % 7 == 0) and i < 52):
                index = near_white(white_x, black_boxes)
                top_box = (white_x + 1, 0, black_boxes[index][0] - white_x - 1, 1.1 * black_boxes[index][3])
                bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
                total_top.append(top_box)
                total_bottom.append(bottom_box)

            elif (i == 4 or ((i - 4) % 7 == 0) and i < 52) or (i == 7 or ((i - 7) % 7 == 0) and i < 52) or (i == 8 or ((i - 8) % 7 == 0) and i < 52):
                index = near_white(white_x, black_boxes)
                index = min(len(black_boxes)-2,index)
                top_box = (black_boxes[index][0]+black_boxes[index][2], 0, black_boxes[index+1][0] - (black_boxes[index][0]+black_boxes[index][2]) - 1, 1.1 * black_boxes[index][3])
                bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
                total_top.append(top_box)
                total_bottom.append(bottom_box)
            elif (i == 5 or ((i - 5) % 7 == 0) and i < 52) or (i == 9 or ((i - 9) % 7 == 0) and i < 52) or (i == 8 or ((i - 8) % 7 == 0) and i < 52):
                index = near_white(white_x, black_boxes)
                top_box = (black_boxes[index][0]+black_boxes[index][2], 0, white_loc[i] - (black_boxes[index][0]+black_boxes[index][2]) - 1, 1.1 * black_boxes[index][3])
                bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
                total_top.append(top_box)
                total_bottom.append(bottom_box)
                #----最后一个框
            else:
                top_box = (white_x + 1, 0, white_loc[i] - white_x - 1, 1.1 * black_boxes[35][3])
                bottom_box = (white_x + 1, 1.1 * black_boxes[35][3], white_loc[i] - white_x - 1, height - 1.1 * black_boxes[35][3])
                total_top.append(top_box)
                total_bottom.append(bottom_box)

        white_loc = np.array(white_loc,dtype=np.int32)
        black_boxes = np.array(black_boxes,dtype=np.int32)
        total_top = np.array(total_top,dtype=np.int32)
        total_bottom = np.array(total_bottom,dtype=np.int32)
        return  white_loc,black_boxes,total_top,total_bottom
    
    def find_black_boxes(self,ori_img):
        thresh = 125
        while True:
            base_img = ori_img.copy()
            height,width,_ = base_img.shape 
            base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
            base_img = remove_region(base_img)
            _,base_img = cv2.threshold(base_img,thresh,255,cv2.THRESH_BINARY_INV)
        
            black_boxes = self.find_black_keys(base_img)
            black_boxes = sorted(black_boxes,key = lambda x:x[0])
            black_loc = [box[0] for box in black_boxes]
            if len(black_loc)>36:
                thresh-=1
            elif len(black_loc)<36:
                thresh+=1
            else:
                break
            if thresh<90 or thresh>150:
                break
        return black_boxes,black_loc 

    def find_white_loc_old(self,black_loc,black_boxes,width):
        white_loc = []
        black_gap1 = black_loc[3] - black_loc[2]  #--第一个周期区域内的黑键间隔
        ratio = 23.0 / 41
        # ratio = 23.0 / 40
        whitekey_width1 = ratio * black_gap1  
        half_width1 = black_boxes[4][2]    #T1中第四个黑键被均分,从该位置开始算区域起始位置
        keybegin = black_loc[4] + half_width1 / 2.0-7.0 * whitekey_width1
        for i in range(10):
            if int(keybegin + i * whitekey_width1) < 0:
                white_loc.append(1)
            else:
                white_loc.append(keybegin + i * whitekey_width1)
        for i in range(6):  #----剩下的6个循环区域
            axis = 8 + i * 5
            black_gap2 = black_loc[axis] - black_loc[axis - 1]
            whitekey_width2 = ratio * black_gap2 
            half_width2 = black_boxes[axis + 1][2] 
            keybegin1 = black_loc[axis + 1] + float(half_width2 / 2.0) - 5.0 * whitekey_width2
            for j in range(1,8):
                white_loc.append(keybegin1 + j * whitekey_width2)
            if i == 5:  #----最后一次循环将钢琴最后一个白键加上
                if width < int(keybegin1 + 8 * whitekey_width2):
                    white_loc.append(width - 1)
                else:
                    white_loc.append(keybegin1 + 8 * whitekey_width2)
        return white_loc 

    def find_white_loc(self,black_loc):
        white_loc = []
        black_gap1 = black_loc[2] - black_loc[1]
        # w_gap1 = 63.0 / 28 * black_gap1 / 3  #---T1前三个白键的间隔
        w_gap1 = 70.0 / 28 * black_gap1 / 3  #---T1前三个白键的间隔
        w_begin1 = black_loc[1] - 2.0 / 3 * black_gap1
        # w_begin1 = black_loc[1] - 1.0 / 2 * black_gap1
        for i in range(3):
            white_loc.append(w_begin1 + i * w_gap1)

        #----最开始的那两个白键
        for i in range(1,3):
            if int(w_begin1 - i * w_gap1) < 0:
                white_loc.append(1)
            else:
                white_loc.append(w_begin1 - i * w_gap1)

        #----周期内后4个白键
        black_gap2 = black_loc[5] - black_loc[4]
        w_gap2 = 94.0 / 27 * black_gap2 / 4  #---T1前三个白键的间隔
        w_begin2 = black_loc[3] - 16.0 / 27 * black_gap2
        # w_begin2 = black_loc[3] - 13.0 / 27 * black_gap2
        for i in range(4):
            white_loc.append(w_begin2 + i * w_gap2)

        #----后面的那几个周期
        for i in range(6):
            axis1 = 7 + i * 5
            black_gap3 = black_loc[axis1] - black_loc[axis1 - 1]
            w_gap3 = 70.0 / 28 * black_gap3 / 3  #---T1前三个白键的间隔
            w_begin3 = black_loc[axis1 - 1] - 2.0 / 3 * black_gap3
            #----前3个白键
            for j in range(3):
                white_loc.append(w_begin3 + j * w_gap3)

            axis2 = 10 + i * 5
            black_gap4 = black_loc[axis2] - black_loc[axis2 - 1]
            w_gap4 = 94.0 / 27 * black_gap4 / 4  #---T1前三个白键的间隔
            w_begin4 = black_loc[axis2 - 2] - 16.0 / 27 * black_gap4
            #---后4个白键
            if i == 5:   #---最后一个周期把最后两个键也加上
                for j in range(6):
                    white_loc.append(w_begin4 + j * w_gap4)
            else:
                for j in range(4):
                    white_loc.append(w_begin4 + j * w_gap4)
        white_loc.sort()
        return white_loc 

    def find_black_keys(self,base_img):
        contours,_ = cv2.findContours(base_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        black_boxes = []
        height,width = base_img.shape[:2]
        for idx,cnt in enumerate(contours):
            (x,y,w,h) = cv2.boundingRect(cnt)
            if h>height*0.3 and w>4:
                x1,y1,x2,y2 = x,y,x+w,y+h 
                for i in range(y2,y1,-1):
                    count = 0 
                    for j in range(x1,x2):
                        if base_img[i,j]!= 0:
                            count+=1 
                    if count > (x2-x1)*0.5:
                        black_boxes.append((x1,y1,w,i-y1))
                        break 
        if len(black_boxes)!=36:
            ws = [box[2] for box in black_boxes]
            ws = np.array(ws)
            me = np.median(ws)
            for i,wd in enumerate(ws):
                if wd<me*0.5:del black_boxes[i]
        return black_boxes

    def key_loc_paper_data(self,base_img):
        white_loc = []
        total_top = []
        total_bottom = []

        ori_img = base_img.copy()
        height,width,_ = base_img.shape 
        base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
        base_img = remove_region(base_img)
        _,base_img = cv2.threshold(base_img,150,255,cv2.THRESH_BINARY_INV)
        
        black_boxes = self.find_black_keys(base_img)
        black_boxes = sorted(black_boxes,key = lambda x:x[0])
        black_boxes = black_boxes[:25]
        assert len(black_boxes) == 25, 'black number is wrong'
        black_loc = [box[0] for box in black_boxes]

        ratio = 22.9 / 41
        for i in range(5):
            axis = i * 5
            black_gap = black_loc[axis + 2] - black_loc[axis + 1]
            whitekey_width = ratio * black_gap
            black_width = black_boxes[axis + 3][2]
            keybegin = black_loc[axis + 3] + black_width * 0.5 - 5.0 * whitekey_width
            if i == 0:
                white_loc.append(keybegin)
            for j in range(1, 8):
                white_loc.append(keybegin + j * whitekey_width)
            if i == 4:
                if width < int(keybegin + 8 * whitekey_width):
                    white_loc.append(width - 1)
                else:
                    white_loc.append(keybegin + 8 * whitekey_width)

        for j in range(5):
            axis = j * 7 + 1
            black_axis = j * 5
            for x in range(1, 8):
                i = axis + x - 1
                white_x = white_loc[i - 1]
                white_width = white_loc[i] - white_x
                if x == 1:
                    bi = black_axis
                    top_box = (white_x, 0, black_boxes[bi][0] - white_x, 1.1 * black_boxes[bi][3])
                    bottom_box = (white_x, 1.1 * black_boxes[bi][3], white_width, height - 1.1 * black_boxes[bi][3])
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)
                elif x == 2:
                    bi1 = black_axis
                    bi1_width, bi1_height = black_boxes[bi1][2], black_boxes[bi1][3]
                    bi2 = black_axis + 1
                    top_box = (
                    black_boxes[bi1][0] + bi1_width, 0, black_boxes[bi2][0] - black_boxes[bi1][0] - bi1_width,
                    1.1 * bi1_height)
                    bottom_box = (white_loc[i - 1], 1.1 * bi1_height, white_width, height - 1.1 * bi1_height)
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)
                elif x == 3:
                    bi = black_axis + 1
                    bi_width, bi_height = black_boxes[bi][2], black_boxes[bi][3]
                    top_box = (
                    black_boxes[bi][0] + bi_width, 0, white_loc[i] - black_boxes[bi][0] - bi_width, 1.1 * bi_height)
                    bottom_box = (white_loc[i - 1], 1.1 * bi_height, white_width, height - 1.1 * bi_height)
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)
                elif x == 4:
                    bi = black_axis + 2
                    bi_width, bi_height = black_boxes[bi1][2], black_boxes[bi1][3]
                    top_box = (white_loc[i - 1], 0, black_boxes[bi][0] - white_loc[i - 1], 1.1 * bi_height)
                    bottom_box = (white_loc[i - 1], 1.1 * bi_height, white_width, height - 1.1 * bi_height)
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)
                elif x == 5:
                    bi1 = black_axis + 2
                    bi2 = black_axis + 3
                    bi1_width, bi1_height = black_boxes[bi1][2], black_boxes[bi1][3]
                    top_box = (
                    black_boxes[bi1][0] + bi1_width, 0, black_boxes[bi2][0] - black_boxes[bi1][0] - bi1_width,
                    1.1 * bi1_height)
                    bottom_box = (white_loc[i - 1], 1.1 * bi1_height, white_width, height - 1.1 * bi1_height)
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)
                elif x == 6:
                    bi1 = black_axis + 3
                    bi2 = black_axis + 4
                    bi1_width, bi1_height = black_boxes[bi1][2], black_boxes[bi1][3]
                    top_box = (
                    black_boxes[bi1][0] + bi1_width, 0, black_boxes[bi2][0] - black_boxes[bi1][0] - bi1_width,
                    1.1 * bi1_height)
                    bottom_box = (white_loc[i - 1], 1.1 * bi1_height, white_width, height - 1.1 * bi1_height)
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)
                elif x == 7:
                    bi = black_axis + 4
                    bi_width, bi_height = black_boxes[bi][2], black_boxes[bi][3]
                    top_box = (
                    black_boxes[bi][0] + bi_width, 0, white_loc[i] - black_boxes[bi][0] - bi_width, 1.1 * bi_height)
                    bottom_box = (white_loc[i - 1], 1.1 * bi_height, white_width, height - 1.1 * bi_height)
                    total_top.append(top_box)
                    total_bottom.append(bottom_box)

        total_top.append((white_loc[-2], 0, width - white_loc[-2], 1.1 * black_boxes[-1][3]))
        total_bottom.append(
            (white_loc[-2], 1.1 * black_boxes[-1][3], width - white_loc[-2], height - 1.1 * black_boxes[-1][3]))

        white_loc = np.array(white_loc, dtype=np.int32)
        black_boxes = np.array(black_boxes, dtype=np.int32)
        total_top = np.array(total_top, dtype=np.int32)
        total_bottom = np.array(total_bottom, dtype=np.int32)
        return white_loc, black_boxes, total_top, total_bottom


