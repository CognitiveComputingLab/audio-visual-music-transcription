import os 
import sys
import cv2 
import numpy as np 
from config import PROJECT_ROOT,cfg
sys.path.append(os.path.join(PROJECT_ROOT,'piano_utils'))
from models.model_helper import ModelProduct
#from keyboard import KeyBoard
import keyboard
from bwlabel import BwLabel
from seghand import SegHand,detect_hand_by_diffimg
from evaluate import Accuracy
from PIL import Image 
from tools.helper import * 
from IPython import embed
from tqdm import tqdm 
import time 

class VisAmtHelper(object):
    def __init__(self,file_mark,
                      midi_file=None,
                      start_frame=0,
                      fps=25.0,
                      midi_offset=0,
                      white_model = None,
                      black_model = None,
                      music_type='record'):
        self.init_model_load(white_model,black_model) 
        self.init_save_file_dir(file_mark)

        self.midi_file = midi_file 
        self.start_frame = start_frame
        self.fps = fps 
        self.midi_offset = midi_offset 
        
        self.detect_hand = True
        self.post_white = True
        self.post_black = True 
        self.type= music_type
        

    def init_model_load(self,white_model=None,black_model=None):
        #self.keyboard = KeyBoard()
        self.hand_seg = SegHand()
        self.modelproduct = ModelProduct(white_model,black_model)
        self.bwlabel = BwLabel()
       
    def init_save_file_dir(self,file_mark):
        self.base_img_dir,self.diff_img_dir,detect_img_dir,press_img_dir = new_save_dir(cfg.SAVE_IMG_DIR,file_mark)
        self.detect_total_img_dir = detect_img_dir[0] 
        self.detect_white_img_dir = detect_img_dir[1] 
        self.detect_black_img_dir = detect_img_dir[2]

        self.press_white_img_dir = press_img_dir[0]
        self.press_black_img_dir = press_img_dir[1]

        self.w_detectPath = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'pitch_white.txt')
        self.b_detectPath = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'pitch_black.txt')
        self.videoPath = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'{}.mp4'.format(file_mark))

        self.bw_index_dict = black_white_index_dict()

        self.white_prob_map = []
        self.black_prob_map = []
        self.w_probPath = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'prob_white.txt')
        self.b_probPath = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'prob_black.txt')
        
        w_detprob_path = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'det_prob_white.txt')
        b_detprob_path = os.path.join(cfg.SAVE_IMG_DIR,file_mark,'det_prob_black.txt')
        
        self.wout = open(w_detprob_path,'w')
        self.bout =  open(b_detprob_path,'w')

    def eval(self,fps):
        self.frame_result = None
        self.note_result = None 
        if self.midi_file is None:
            return 
        pframe_time = 1.0/fps 
        evaluate = Accuracy(self.midi_file,
                            self.w_detectPath,
                            self.b_detectPath,
                            pframe_time = pframe_time,
                            start_frame = self.start_frame,
                            midi_offset = self.midi_offset)
        
        self.frame_result = evaluate.get_frame_result() 
        self.note_result = evaluate.get_note_result()

    def process_img_dir(self,img_dir):
        img_lists = [os.path.join(img_dir,x) for x in os.listdir(img_dir) 
                if x.endswith('jpg') or x.endswith('png')]
        img_lists.sort()
        base_info = find_base_img(self.keyboard,self.modelproduct,img_lists)
        base_img = base_info['base_img']
        base_all_img = base_info['img']
        keyboard_rect = base_info['rect']
        start_frame = base_info['count_frame']
        warp_M = base_info['warp_M']
        rote_M = base_info['rote_M']
        if base_img is None:
            return  
        white_loc,black_boxes,total_top,total_bottom = find_key_loc(self.bwlabel,base_img)
        save_img = vis_white_loc_boxes(base_img,white_loc,black_boxes)
        cv2.imwrite(os.path.join(self.base_img_dir,'base.jpg'),save_img)
        cv2.imwrite(os.path.join(self.base_img_dir,'base_ori.jpg'),base_all_img)
        self.count_frame = 0  
        avgtimer = timer()
        handtimer = timer()
        keypresstimer = timer()
        vistimer = timer()
        fps = self.fps 
        fwhite = open(self.w_detectPath,'w')
        fblack = open(self.b_detectPath,'w')

        for img_file in tqdm(img_lists):
            avgtimer.tic()
            img = Image.open(img_file)
            w,h = img.size 
            self.count_frame+=1 
            if self.count_frame==start_frame:
                continue 
            file_seq = os.path.basename(img_file).split('.')[0]
            opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            if rote_M is not None:
            	opencv_img = cv2.warpAffine(opencv_img,rote_M,(w,h))
            if warp_M is not None:
            	opencv_img = cv2.warpPerspective(opencv_img,warp_M,(w,h))
            img = Image.fromarray(cv2.cvtColor(opencv_img,cv2.COLOR_BGR2RGB))
            handtimer.tic()
            if self.detect_hand:
                hand_boxes,mask = self.hand_seg.segment_detect_hand(img,keyboard_rect)
            else:
                hand_boxes,tmp = detect_hand_by_diffimg(base_all_img,opencv_img,keyboard_rect)
            
            if self.type=='paper':
                has_hand,tmp = detect_hand_by_diffimg(base_all_img,opencv_img,keyboard_rect)
            else:
                has_hand = hand_boxes 
            cur_keyboard_img = opencv_img[keyboard_rect[1]:keyboard_rect[3], keyboard_rect[0]:keyboard_rect[2]]
            handetime = handtimer.toc()

            if len(has_hand) == 0:
                base_img = update_base_img(base_img,cur_keyboard_img,white_loc,hand_boxes)
                base_all_img = opencv_img.copy()
                continue
            #cv2.imwrite(os.path.join(self.base_img_dir,os.path.basename(img_file)),base_img)
            keypresstimer.tic()
            if self.detect_hand:
                cur_keyboard_mask = mask[keyboard_rect[1]:keyboard_rect[3], keyboard_rect[0]:keyboard_rect[2]]
            else:
                cur_keyboard_mask = None
            white_keys_list, black_keys_list, diff_img = self.keypress(base_img,
                                                                       cur_keyboard_img,
                                                                       cur_keyboard_mask,
                                                                       black_boxes,
                                                                       white_loc,
                                                                       hand_boxes,
                                                                       file_seq)
            keyetime = keypresstimer.toc()
            vistimer.tic()
            '''
            save_white_img = vis_detect_white_key(opencv_img,
                                                  hand_boxes,
                                                  white_keys_list,
                                                  white_loc,
                                                  keyboard_rect,
                                                  total_top,
                                                  total_bottom)

            save_black_img = vis_detect_black_key(opencv_img,
                                                  hand_boxes,
                                                  black_keys_list,
                                                  keyboard_rect,
                                                  black_boxes)
            '''
            save_total_img = vis_detect_total_key(opencv_img,
                                                  os.path.basename(img_file),
                                                  hand_boxes,
                                                  white_keys_list,
                                                  black_keys_list,
                                                  white_loc,
                                                  keyboard_rect,
                                                  total_top,
                                                  total_bottom,
                                                  black_boxes)
            diff_img = vis_diff_img_key(diff_img,os.path.basename(img_file),
                                        hand_boxes,white_loc,black_boxes,keyboard_rect)
            # cv2.imwrite(os.path.join(self.detect_white_img_dir,os.path.basename(img_file)),save_white_img)
            # cv2.imwrite(os.path.join(self.detect_black_img_dir,os.path.basename(img_file)),save_black_img)
            cv2.imwrite(os.path.join(self.detect_total_img_dir,os.path.basename(img_file)),save_total_img)
            cv2.imwrite(os.path.join(self.diff_img_dir,os.path.basename(img_file)),diff_img)
            save_detect_result(white_keys_list, img_file, fwhite, fps)
            save_detect_result(black_keys_list, img_file, fblack, fps)
            avgtimer.toc()
            #print('process {} total cost:{:.3}s || hand detect:{:.3}s || keypress:{:.3}s || vis:{:.3}s'.format(os.path.basename(img_file),
            #                                                                    avgtimer.toc(), handetime, keyetime,vistimer.toc()))
            if self.count_frame%5==0 and cfg.UPDATE_BACKGROUND:
                base_img = update_base_img(base_img,cur_keyboard_img,white_loc,hand_boxes)
        fwhite.close()
        fblack.close()
        print('avg process time:{:.3}s'.format(avgtimer.elapsed()))
        self.eval(fps)
        #img2video(self.detect_total_img_dir,self.videoPath)
        save_prob_file(self.w_probPath,self.b_probPath,self.white_prob_map,self.black_prob_map)
        self.wout.close()
        self.bout.close()


    def process_video(self,video_file):
        capture = cv2.VideoCapture(video_file)
        if not capture.isOpened():
            raise ValueError('read video wrong')
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        base_info = find_video_base_img(self.keyboard,self.modelproduct,video_file)
        base_img = base_info['base_img']
        keyboard_rect = base_info['rect']
        start_frame = base_info['count_frame']
        warp_M = base_info['warp_M']
        rote_M = base_info['rote_M']
        if base_img is None:
            return 
        white_loc,black_boxes,total_top,total_bottom = find_key_loc(self.bwlabel,base_img)
        save_img = vis_white_loc_boxes(base_img,white_loc,black_boxes)
        cv2.imwrite(os.path.join(self.base_img_dir,'base.jpg'),save_img)
        self.count_frame = 0
        avgtimer = timer()
        handtimer = timer()
        keypresstimer = timer()
        fwhite = open(self.w_detectPath,'w')
        fblack = open(self.b_detectPath,'w')
        for idx in tqdm(range(total_frames)):
            avgtimer.tic()
            ret,opencv_img = capture.read()
            h,w,_ = opencv_img.shape
            self.count_frame+=1 
            if self.count_frame==start_frame:
                continue 
            if not ret:
                break 
            if rote_M is not None:
                opencv_img = cv2.warpAffine(opencv_img,rote_M,(w,h))
            if warp_M is not None:
                opencv_img = cv2.warpPerspective(opencv_img,warp_M,(w,h))
            img = Image.fromarray(cv2.cvtColor(opencv_img,cv2.COLOR_BGR2RGB))            
            handtimer.tic()
            cur_keyboard_img = opencv_img[keyboard_rect[1]:keyboard_rect[3], keyboard_rect[0]:keyboard_rect[2]]
            hand_boxes, mask = self.hand_seg.segment_detect_hand(img, keyboard_rect)
            handetime = handtimer.toc()
            if len(hand_boxes) == 0:
                base_img = update_base_img(base_img,cur_keyboard_img,white_loc,hand_boxes)
                continue
            file_seq = str(self.count_frame).zfill(5)
            keypresstimer.tic()
            cur_keyboard_mask = mask[keyboard_rect[1]:keyboard_rect[3], keyboard_rect[0]:keyboard_rect[2]]
            white_keys_list, black_keys_list, diff_img = self.keypress(base_img,
                                                                       cur_keyboard_img,
                                                                       cur_keyboard_mask,
                                                                       black_boxes,
                                                                       white_loc,
                                                                       hand_boxes,
                                                                       file_seq)
            keyetime = keypresstimer.toc()
            avgtimer.toc()
            #print('process {}.jpg total cost:{:.3}s || hand detect:{:.3}s || keypress:{:.3}s'.format(str(self.count_frame).zfill(5),
            #                                                                                avgtimer.toc(), handetime, keyetime))
            save_white_img = vis_detect_white_key(opencv_img,
                                                  hand_boxes,
                                                  white_keys_list,
                                                  white_loc,
                                                  keyboard_rect,
                                                  total_top,
                                                  total_bottom)
            save_black_img = vis_detect_black_key(opencv_img,
                                                  hand_boxes,
                                                  black_keys_list,
                                                  keyboard_rect,
                                                  black_boxes)

            save_total_img = vis_detect_total_key(opencv_img,
                                                  '{}.jpg'.format(str(self.count_frame).zfill(5)),
                                                  hand_boxes,
                                                  white_keys_list,
                                                  black_keys_list,
                                                  white_loc,
                                                  keyboard_rect,
                                                  total_top,
                                                  total_bottom,
                                                  black_boxes)
            cv2.imwrite(os.path.join(self.detect_white_img_dir, '{}.jpg'.format(str(self.count_frame).zfill(5))),save_white_img)
            cv2.imwrite(os.path.join(self.detect_black_img_dir, '{}.jpg'.format(str(self.count_frame).zfill(5))),save_black_img)
            cv2.imwrite(os.path.join(self.detect_total_img_dir, '{}.jpg'.format(str(self.count_frame).zfill(5))),save_total_img)
            cv2.imwrite(os.path.join(self.diff_img_dir, '{}.jpg'.format(str(self.count_frame).zfill(5))), diff_img)
            save_detect_result(white_keys_list, '{}.jpg'.format(str(self.count_frame).zfill(5)), fwhite, fps)
            save_detect_result(black_keys_list, '{}.jpg'.format(str(self.count_frame).zfill(5)), fblack, fps)
        print('avg process time:{:.3}s'.format(avgtimer.elapsed()))
        fwhite.close()
        fblack.close()
        capture.release() 
        self.eval(fps)
        img2video(self.detect_total_img_dir,self.videoPath)

    def keypress(self,
                base_img,
                cur_img,
                hand_mask,
                black_boxes,
                white_loc,
                hand_boxes,
                file_seq):
        index_list = near_white(white_loc,hand_boxes)
        cur_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2GRAY)
        base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
        diff_img = cv2.absdiff(base_img,cur_img)
        diff_img = cv2.cvtColor(diff_img,cv2.COLOR_GRAY2BGR)
        
        h,w= diff_img.shape[:2]
        base_img1 = base_img.astype(np.float32)
        cur_img1 = cur_img.astype(np.float32)
        diff_img_white = np.maximum(base_img1-cur_img1,0).astype(np.uint8)
        diff_img_black = np.maximum(cur_img1-base_img1,0).astype(np.uint8)
        offset = 3
        whole_list = []
        for hand_list in  index_list:
            for index in range(hand_list[0],hand_list[1]+1):
                whole_list.append(index)
        whole_list = np.maximum(whole_list,0)
        whole_list = np.minimum(whole_list,len(white_loc)-2) 
        whole_list = list(set(whole_list))
        whole_list.sort()
        detect_white_keys = []
        white_index_prob_map = dict() 
        
        wprob_map = [0 for x in range(53)]
        bprob_map = [0 for x in range(36)] 

        self.wout.write('{}.jpg\n'.format(file_seq))
        self.bout.write('{}.jpg\n'.format(file_seq))

        if len(whole_list)>0:
            input_imgs = list()
            for index in whole_list:
                start = max(int(white_loc[index]-offset),0)
                end = min(int(white_loc[index+1]+offset),w)
                crop_img = diff_img[0:h,start:end]
                input_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
                press_path = os.path.join(self.press_white_img_dir,'{}_{}.jpg'.format(file_seq,index+1))
                cv2.imwrite(press_path,crop_img)
                input_imgs.append(input_img)
            stime = time.time()
            pred,prob = self.modelproduct.detect_white_keys(input_imgs)
            
            for idx,key_index in enumerate(whole_list):
                data = '{}\t{:.3}\n'.format(key_index+1,prob[idx])
                self.wout.write(data)
                wprob_map[key_index] = prob[idx]
                if pred[idx]==1:
                    if self.post_white:
                        press,flag = vertify_press_white(key_index+1,hand_mask,self.bw_index_dict,black_boxes,hand_boxes,white_loc,prob[idx])
                        if flag==0 and press:
                            detect_white_keys.append(key_index+1)
                            white_index_prob_map[key_index+1] = prob[idx]
                        if flag:
                            start = max(int(white_loc[key_index]-offset),0)
                            end = min(int(white_loc[key_index+1]+offset),w)
                            crop_img = diff_img_white[0:h,start:end]
                            input_img = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
                            input_imgs = [input_img]
                            cur_pred,cur_prob = self.modelproduct.detect_white_keys(input_imgs)
                            if cur_pred[0]==1:
                                detect_white_keys.append(key_index+1)
                                white_index_prob_map[key_index+1] = prob[idx]
                    else:
                        detect_white_keys.append(key_index+1)
                        white_index_prob_map[key_index+1] = prob[idx]

        black_index_list = near_black(black_boxes,hand_boxes)
        black_whole_list = []
        for hand_list in black_index_list:
            for index in range(hand_list[0],hand_list[1]+1):
                black_whole_list.append(index)
        black_whole_list = np.maximum(black_whole_list,0)
        black_whole_list = np.minimum(black_whole_list,len(black_boxes)-1) 
        black_whole_list = list(set(black_whole_list))
        black_whole_list.sort()
        detect_black_keys = []
        black_index_prob_map = dict()
        if len(black_whole_list)>0:
            input_imgs = list()
            for index in black_whole_list:
                bbox = black_boxes[index]
                x1,y1,x2,y2 = bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]
                x1,x2,y2 = max(0,x1-offset),min(w,x2+offset),min(h,y2+offset-1)
                crop_img = diff_img[y1:y2,x1:x2]
                img_ = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))
                input_imgs.append(img_)
                press_path = os.path.join(self.press_black_img_dir,'{}_{}.jpg'.format(file_seq,index+1))
                cv2.imwrite(press_path,crop_img) 
            pred,prob = self.modelproduct.detect_black_keys(input_imgs)

            for idx,key_index in enumerate(black_whole_list):
                data = '{}\t{:.3}\n'.format(key_index+1,prob[idx])
                self.bout.write(data)
                bprob_map[key_index] = prob[idx]
                if pred[idx]==1: 
                    if self.post_black:
                        if vertify_press_black(key_index,hand_mask,black_boxes):
                            detect_black_keys.append(key_index+1)
                            black_index_prob_map[key_index+1] = prob[idx]
                    else:
                        detect_black_keys.append(key_index+1)
                        black_index_prob_map[key_index+1] = prob[idx]

        for wkey,wprob in white_index_prob_map.items():
            bkeys = self.bw_index_dict[str(wkey)]
            for bkey in bkeys:
                if bkey in black_index_prob_map.keys():
                    bprob = black_index_prob_map[bkey]
                    if (wprob-bprob)>0.15 or (wprob>0.93 and bprob<0.9):
                        detect_black_keys.remove(bkey)
                    if bprob>0.96 and wprob<0.9:
                        detect_white_keys.remove(wkey)
                        break 
        self.white_prob_map.append(wprob_map)
        self.black_prob_map.append(bprob_map)
        return detect_white_keys,detect_black_keys,diff_img  
