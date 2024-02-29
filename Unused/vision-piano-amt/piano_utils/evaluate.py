#-*- coding:utf-8 -*-
import cv2
import numpy
import argparse
import os
import shutil
import mido
from IPython import embed
import sys 
sys.path.append(os.path.abspath('.'))
from config import cfg 

class Accuracy(object):
    def __init__(self, 
                midiPath=None, 
                w_detectPath=None, 
                b_detectPath=None,
                pframe_time=0.04,
                start_frame = 0,
                midi_offset = 1.5,
                frameOffset=2, 
                tolerance=0.15):
        self.midiPath = midiPath
        self.w_detectPath = w_detectPath
        self.b_detectPath = b_detectPath
        self.pframe_time = pframe_time 
        self.offTime = start_frame * pframe_time

        self.frameOffset = frameOffset  #前面两帧没出现则认为是新的note
        self.tolerance = tolerance  #误差计算误差为2s
        self.tol_frame = 0
        self.midi_offset = midi_offset 

        self.black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                          31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                          60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
        self.white_num = [x for x in range(1, 89) if x not in self.black_num]

        self.run()
    
    def run(self):
        if self.midiPath is not None:
            if self.midiPath.endswith('txt'):
                self.pitch_onset = self.processMidibytxt(self.midiPath,self.offTime,self.midi_offset) 
            else:
                self.pitch_onset = self.processMidi(self.midiPath,self.offTime,self.midi_offset)
            evalfile = os.path.join(os.path.dirname(self.w_detectPath),'evalresult.txt') 
            self.evalout = open(evalfile,'w')
            #self.black_precision()
            #self.white_precision()

            self.evaluate_frame_precision()
            self.Total_precision()
            self.save_midi()
            self.pitch2note()
            self.evalout.close()
        else:
            self.b_pro_onset=self.processDetect(self.b_detectPath,mode='black')
            self.w_pro_onset = self.processDetect(self.w_detectPath) 
            self.pitch2note()

    def get_frame_result(self):
        result = {
                'black':{'precies':self.b_precies,'recall':self.b_recall,'F':self.b_F},
                'white':{'precies':self.w_precies,'recall':self.w_recall,'F':self.w_F}
        }
        return result 

    def get_note_result(self):
        result = {
                'black':{'precies':self.noteb_precies,'recall':self.noteb_recall,'F':self.noteb_F},
                'white':{'precies':self.notew_precies,'recall':self.notew_recall,'F':self.notew_F}
        }
        return result 
    
    def processMidibytxt(self,midiPath,offTime=0,midi_offset=0):
        with open(midiPath,'r') as fr:
            items = fr.readlines()
        self.pitch_onset_offset = []
        pitch_onset = []
        for item in items:
            item = item.strip().split('\t')
            po = [float(item[0])-midi_offset+offTime,float(item[1])-midi_offset+offTime,int(item[2])]
            self.pitch_onset_offset.append(po)
            pitch_onset.append([float(item[0])-midi_offset+offTime,int(item[2])])
        
        pitch_onset = sorted(pitch_onset, key=lambda x: (x[0], x[1]))
        self.pitch_onset_offset = sorted(self.pitch_onset_offset,key=lambda x:(x[0],x[1],x[2]))
        return pitch_onset 

    def processMidi(self, midiPath, offTime, midi_offset):
        mid = mido.MidiFile(midiPath)
        timeCount = 0
        dataList = []
        for msg in mid:
            if not msg.is_meta:
                if msg.type == 'control_change':
                    timeCount = timeCount + msg.time
                elif msg.type == 'note_on' or msg.type == 'note_off':
                    timeCount = timeCount + msg.time
                    if 'wmv' in self.midiPath:
                        data = [msg.type, msg.note - 8, msg.velocity, timeCount]
                    else:
                        data = [msg.type, msg.note - 20, msg.velocity, timeCount]
                    dataList.append(data)
        dict1 = {}
        result = []
        for data in dataList:
            if data[0] == 'note_on' and data[2] > 0:
                dict1[data[1]] = data[1:]
            else:
                dict1[data[1]].append(data[3])
                result.append(dict1.pop(data[1]))
        result = sorted(result, key = lambda x : x[2])
        pitch_onset = []
        for item in result:
            po = [item[2] - midi_offset + offTime, item[0]]
            pitch_onset.append(po)
        pitch_onset = sorted(pitch_onset, key=lambda x: (x[0], x[1]))
        #print(pitch_onset) 
        self.pitch_onset_offset = []
        for item in result:
            po = [item[2] - midi_offset + offTime, item[3] - midi_offset + offTime,item[0]]
            self.pitch_onset_offset.append(po)
        self.pitch_onset_offset = sorted(self.pitch_onset_offset,key=lambda x:(x[0],x[1],x[2]))
        return pitch_onset
    
    def save_midi(self):
        whitepath = os.path.join(os.path.dirname(self.w_detectPath),'midi_white.txt')
        blackpath = os.path.join(os.path.dirname(self.w_detectPath),'midi_black.txt')

        pitch_white = [[line[0],self.white_num.index(line[1]) + 1]
                for line in self.pitch_onset if line[1] in self.white_num]

        pitch_balck = [[line[0],self.black_num.index(line[1]) + 1]
                        for line in self.pitch_onset if line[1] in self.black_num]
        
        with open(whitepath,'w') as fout:
            for po in pitch_white:
                count_frame = int(po[0]/self.pframe_time)
                dat = 'frame{} {:.5} {}\n'.format(count_frame,po[0],po[1])
                fout.write(dat)

        with open(blackpath,'w') as fout:
            for po in pitch_balck:
                count_frame = int(po[0]/self.pframe_time)
                dat = 'frame{} {:.5} {}\n'.format(count_frame,po[0],po[1])
                fout.write(dat)
    
    def pitch2note(self):
        whitepath = os.path.join(os.path.dirname(self.w_detectPath),'note_white.txt')
        blackpath = os.path.join(os.path.dirname(self.w_detectPath),'note_black.txt')
        with open(whitepath,'w') as fout:
            for po in self.w_pro_onset:
                count_frame = int(po[0]/self.pframe_time)
                dat = 'frame{} {:.5} {:.5} {}\n'.format(count_frame,po[0],po[2],po[1])
                fout.write(dat)

        with open(blackpath,'w') as fout:
            for po in self.b_pro_onset:
                count_frame = int(po[0]/self.pframe_time)
                dat = 'frame{} {:.5} {:.5} {}\n'.format(count_frame,po[0],po[2],po[1])
                fout.write(dat)

    def processDetect(self, detectPath,mode='white'):
        with open(detectPath, 'r') as f:
            lines = f.readlines()
        filenames = []
        times = []
        keys = []
        for i, line in enumerate(lines):
            key = []
            line = line.strip().split()
            filenames.append(line[0])
            frame = int(os.path.basename(line[0]).split('.')[0])
            #times.append(float(line[1]))
            times.append(float(frame*self.pframe_time))
            for j in range(2, len(line)):
                key.append(int(line[j]))
            keys.append(key)

        pro_onset = []
        #-----txt中第一帧就记录了按键
        for i in range(1,len(times)):
            current_keys = set(keys[i])
            last_keys = set(keys[i - 1])
            difSet = [x for x in list(current_keys.symmetric_difference(last_keys)) if not (x == 0 or x in last_keys)]

            if (len(difSet) > 0):
                final_set = difSet[:]
                for key in difSet:
                    if (i - self.frameOffset) > 0:  
                        for m in range(2, self.frameOffset + 1):
                            if key in keys[i - m]:
                                final_set = [x for x in final_set if x is not key]
                                break       
                    else:
                        for m in range(i):   #---对于视频的前m帧,只需要判断前面几帧是否有重复
                            if key in keys[m]:
                                final_set = [x for x in final_set if x is not key]
                                break

                final_set = sorted(final_set)
                for pressed_key in final_set:
                    count = 0
                    end_frame = i+1
                    for j in range(i+1,len(times)):
                        if pressed_key in keys[j]:
                            count+=1
                            end_frame = j 
                        else:break 
                    if mode=='white':
                        if count>=1:
                            data = [times[i], pressed_key,times[end_frame]]
                            pro_onset.append(data)
                    else:
                        if count>=1:
                            data = [times[i], pressed_key,times[end_frame]]
                            pro_onset.append(data)
        if len(keys)>0 and len(pro_onset)>0:
            for note in keys[0]:
                count = 0
                end_frame = 1
                if note==0 or note==pro_onset[0][1]:continue
                for j in range(1,len(times)):
                    if note in keys[j]:
                        count+=1
                        end_frame = j
                    else:break 
                if count>=1:
                    data = [times[0],note,times[end_frame]]
                    pro_onset.append(data)
        if len(pro_onset) > 0:
            pro_onset = sorted(pro_onset, key=lambda x: (x[0], x[1],x[2]))
        return pro_onset
    
    def cal_F(self,recall,precise):
        if recall==0 or precise==0:
            return 0.0 
        F = 2.0*recall*precise/(recall+precise)
        return F

    def cuont_acu(self, pitch_onset, pro_onset):
        right_keys_precies = []
        right_keys_recall = []
        if len(pitch_onset) == 0 or len(pro_onset) == 0:
            return 0.0, 0.0, 0.0,0.0
        else:
            for i, w_key in enumerate(pitch_onset):
                cur_time = w_key[0]
                cur_key = w_key[1]
                cur_matchs = []
                for pro_key in pro_onset:
                    if (abs(cur_time - pro_key[0]) < self.tolerance):
                        if (cur_key == pro_key[1]):
                            cur_matchs.append(pro_key)
                if len(cur_matchs)>0:
                    right_keys_recall.append(cur_matchs)
                else:
                    count_frame = int(w_key[0]/self.pframe_time)
                    #print(count_frame,w_key[1])
            for i, w_key in enumerate(pro_onset):
                cur_time = w_key[0]
                cur_key = w_key[1]
                cur_matchs = []
                for pitch_key in pitch_onset:
                    if (abs(cur_time - pitch_key[0]) < self.tolerance):
                        if (cur_key == pitch_key[1]):
                            cur_matchs.append(pitch_key)
                if len(cur_matchs)>0:
                    right_keys_precies.append(cur_matchs)
                else:
                    count_frame = int(w_key[0]/self.pframe_time)
                    #print(count_frame,w_key[1])
   
            #right_keys1 = sorted(right_keys1, key=lambda x: (x[0],x[1]))
            #right_keys1=list(set([tuple(t) for t in right_keys1]))
            #right_keys1 = sorted(right_keys1, key=lambda x: (x[0],x[1]))
            conf1 = len(right_keys_precies) / len(pro_onset)
            conf2 = len(right_keys_recall) / len(pitch_onset)
            return (conf1, conf2, len(right_keys_precies), len(right_keys_recall))


    def black_precision(self):
        self.b_pro_onset=self.processDetect(self.b_detectPath,mode='black')
        self.b_pro_onset = [po for po in self.b_pro_onset if po[1]!=0]
        pitch_balck = [[line[0],self.black_num.index(line[1]) + 1]
                        for line in self.pitch_onset if line[1] in self.black_num] 
        self.noteb_precies, self.noteb_recall, B_keys_pres,B_keys_recall = self.cuont_acu(pitch_balck, self.b_pro_onset)
        self.noteb_F = self.cal_F(self.noteb_recall,self.noteb_precies)
        data = 'note black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(self.noteb_precies,self.noteb_recall,self.noteb_F)
        self.evalout.write(data+'\n')
        print(data)
        return self.noteb_precies,self.noteb_recall,B_keys_pres,B_keys_recall 

    def white_precision(self):
        self.w_pro_onset = self.processDetect(self.w_detectPath) 
        self.w_pro_onset = [po for po in self.w_pro_onset if po[1]!=0]
        pitch_white = [[line[0],self.white_num.index(line[1]) + 1]
                        for line in self.pitch_onset if line[1] in self.white_num]
        self.notew_precies, self.notew_recall, W_keys_pres,W_keys_recall = self.cuont_acu(pitch_white, self.w_pro_onset)
        self.notew_F = self.cal_F(self.notew_recall,self.notew_precies)
        data = 'note white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(self.notew_precies,self.notew_recall,self.notew_F)
        self.evalout.write(data+'\n')
        print(data)
        return self.notew_precies,self.notew_recall,W_keys_pres,W_keys_recall 

    def Total_precision(self):
        self.w_pro_onset = self.processDetect(self.w_detectPath)
        self.b_pro_onset = self.processDetect(self.b_detectPath,mode='black')
        Totalkeys = self.w_pro_onset[:]
        for line in self.b_pro_onset:
            Totalkeys.append(line)
        Prokeys_nums = len(self.w_pro_onset) + len(self.b_pro_onset)
        Pitchkeys_num = len(self.pitch_onset)
        _, _, B_keys_pres,B_keys_recall = self.black_precision()
        _, _, W_keys_pres,W_keys_recall = self.white_precision()
        RightKeys_pres = B_keys_pres+W_keys_pres
        RightKeys_recall = B_keys_recall+W_keys_recall 

        conf1 = RightKeys_pres / Prokeys_nums 
        conf2 = RightKeys_recall / Pitchkeys_num
        F = self.cal_F(conf1,conf2)
        data = 'note total\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(conf1,conf2,F)
        self.evalout.write(data)
        #print(data)
        return conf1, conf2
    
    
    def evaluate_frame_precision(self):
        def parse_detect_file(detectPath):
            with open(detectPath, 'r') as f:
                lines = f.readlines()
            frames = []
            keys = []
            for i, line in enumerate(lines):
                line = line.strip().split()
                frame = int(os.path.basename(line[0]).split('.')[0])
                for j in range(2, len(line)):
                    if int(line[j])>0:
                        keys.append(int(line[j]))
                        frames.append(frame) 
            return frames,keys 

        def cal_acc(det_frames,det_keys,midi_frames,midi_keys):
            recall_count = 0 
            precise_count = 0
            #---相当于是计算帧级的准确率
            for idx,mframe in enumerate(midi_frames):
                match = False 
                for idy,dframe in enumerate(det_frames):
                    if abs(mframe-dframe)<=self.tol_frame and det_keys[idy]==midi_keys[idx]:  
                        match = True 
                if match:recall_count+=1
            for idx,dframe in enumerate(det_frames):
                match = False 
                for idy,mframe in enumerate(midi_frames):
                    if abs(mframe-dframe)<=self.tol_frame and det_keys[idx]==midi_keys[idy]:
                        match = True 
                if match:precise_count+=1

            precise,recall = 0.0,0.0
            if len(det_frames)>0:
                precise = precise_count/len(det_frames)
            if len(midi_frames)>0:
                recall = recall_count/len(midi_frames)
            print(precise_count,len(det_frames),recall_count,len(midi_frames))
            F = self.cal_F(recall,precise)
            return precise,recall,F 

        midi_white_frames,midi_white_keys = [],[]
        midi_black_frames,midi_black_keys = [],[]
        for pof in self.pitch_onset_offset:
            start_frame = round(pof[0]/self.pframe_time)
            end_frame = round(pof[1]/self.pframe_time)
            
            if pof[2] in self.black_num:
                for frame in range(start_frame,end_frame):
                    midi_black_frames.append(frame)
                    midi_black_keys.append(self.black_num.index(pof[2])+1)
            else:
                for frame in range(start_frame,end_frame):
                    midi_white_frames.append(frame)
                    midi_white_keys.append(self.white_num.index(pof[2])+1)

        det_white_frames,det_white_keys = parse_detect_file(self.w_detectPath)
        det_black_frames,det_black_keys = parse_detect_file(self.b_detectPath)
        
        self.w_precies,self.w_recall,self.w_F = cal_acc(det_white_frames,det_white_keys,midi_white_frames,midi_white_keys)
        self.b_precies,self.b_recall,self.b_F = cal_acc(det_black_frames,det_black_keys,midi_black_frames,midi_black_keys)

        data = 'frame black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(self.b_precies,self.b_recall,self.b_F)
        print(data) 
        self.evalout.write(data+'\n')

        data = 'frame white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(self.w_precies,self.w_recall,self.w_F)
        print(data) 
        self.evalout.write(data+'\n')
        
      

if __name__ == '__main__':
    #eval_record = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']

    #eval_record = ['left_140','left_280','left_400','left_510','left_600']
    #eval_record = ['middle_140','middle_280','middle_400','middle_510','middle_600']
    #eval_record = ['right_140','right_280','right_400','right_510','right_600']

    #eval_record = ['1_baseline','1_right_260','1_right_400','1_right_520','1_right_630','1_right_730']
    #eval_record = ['2_baseline','2_right_280','2_right_400','2_right_520','2_right_630','2_right_730']
    #eval_record = ['3_middle_260','3_middle_400','3_middle_530','3_middle_690','3_middle_800']
    #eval_record = ['4_left_240','4_left_390','4_left_520','4_left_620','4_left_730']
    #eval_record = ['middle_140','middle_280','middle_400','middle_510','middle_600','3_middle_260','3_middle_400','3_middle_530','3_middle_690','3_middle_800']
    eval_record = ['level_1_no_2']
    W_frame_recall,W_frame_precies,W_frame_F = 0,0,0
    B_frame_recall,B_frame_priecies,B_frame_F = 0,0,0
    W_note_recall,W_note_precies,W_note_F = 0,0,0
    B_note_recall,B_note_precies,B_note_F = 0,0,0

    for rec in eval_record:
        if rec in cfg.EVALUATE_MAP.keys():
            print(rec)
            w_detectPath = '/home/data/lj/Piano/saved/network/{}/pitch_white.txt'.format(rec)
            b_detectPath = '/home/data/lj/Piano/saved/network/{}/pitch_black.txt'.format(rec)
            if not os.path.exists(w_detectPath):continue 
            midiPath = cfg.EVALUATE_MAP[rec]['midi']
            fps = cfg.EVALUATE_MAP[rec]['fps']
            midi_offset = cfg.EVALUATE_MAP[rec]['midi_offset']
            start_frame = cfg.EVALUATE_MAP[rec]['start_frame']
            Acu = Accuracy(midiPath, w_detectPath, b_detectPath,start_frame=start_frame,pframe_time=1/fps,midi_offset=midi_offset)
            frame_result = Acu.get_frame_result()
            note_result = Acu.get_note_result()
            wf_rec,wf_prec,wf_F = frame_result['white']['recall'],frame_result['white']['precies'],frame_result['white']['F']
            bf_rec,bf_prec,bf_F = frame_result['black']['recall'],frame_result['black']['precies'],frame_result['black']['F']
            wn_rec,wn_prec,wn_F = note_result['white']['recall'],note_result['white']['precies'],note_result['white']['F']
            bn_rec,bn_prec,bn_F = note_result['black']['recall'],note_result['black']['precies'],note_result['black']['F']
            W_frame_recall+=wf_rec 
            W_frame_precies+=wf_prec 
            W_frame_F+=wf_F 
            B_frame_recall+=bf_rec 
            B_frame_priecies+=bf_prec 
            B_frame_F+=bf_F 
            W_note_recall+=wn_rec 
            W_note_precies+=wn_prec 
            W_note_F+=wn_F 
            B_note_recall+=bn_rec 
            B_note_precies+=bn_prec 
            B_note_F+=bn_F 
    img_len = len(eval_record)
    W_frame_recall,W_frame_precies,W_frame_F = W_frame_recall/img_len,W_frame_precies/img_len,W_frame_F/img_len 
    B_frame_recall,B_frame_priecies,B_frame_F = B_frame_recall/img_len,B_frame_priecies/img_len,B_frame_F/img_len 
    W_note_recall,W_note_precies,W_note_F = W_note_recall/img_len,W_note_precies/img_len,W_note_F/img_len 
    B_note_recall,B_note_precies,B_note_F = B_note_recall/img_len,B_note_precies/img_len,B_note_F/img_len 
    print('avg frame black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(B_frame_priecies,B_frame_recall,B_frame_F))
    print('avg frame white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(W_frame_precies,W_frame_recall,W_frame_F))
    print('avg note black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(B_note_precies,B_note_recall,B_note_F))
    print('avg note white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(W_note_precies,W_note_recall,W_note_F))
    
    path = '/home/data/lj/Piano/saved/network'
    dirs = [os.path.join(path,x) for x in os.listdir(path)]
    for cur_dir in dirs:
        w_detectPath = os.path.join(cur_dir,'pitch_white.txt')
        b_detectPath = os.path.join(cur_dir,'pitch_black.txt')
        if not os.path.exists(w_detectPath):continue 
        acc = Accuracy(w_detectPath=w_detectPath,b_detectPath=b_detectPath)

