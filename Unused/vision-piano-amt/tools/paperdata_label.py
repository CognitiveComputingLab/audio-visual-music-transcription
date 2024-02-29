import os 
import shutil 
import cv2
from PIL import Image 
from IPython import embed 
import numpy as np 

TRAIN_ITEMS = [
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/1_1.txt','img_dir':'/home/data/lj/Piano/saved/1_1_img','start_frame':70},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/1_2.txt','img_dir':'/home/data/lj/Piano/saved/1_2_img','start_frame':89},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/1_3.txt','img_dir':'/home/data/lj/Piano/saved/1_3_img','start_frame':78},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/1_4.txt','img_dir':'/home/data/lj/Piano/saved/1_4_img','start_frame':52},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/2_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/2_+45_1_img','start_frame':85},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/2_0_1.txt','img_dir':'/home/data/lj/Piano/saved/2_0_1_img','start_frame':104},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/2_0_2.txt','img_dir':'/home/data/lj/Piano/saved/2_0_2_img','start_frame':49},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/2_0_3.txt','img_dir':'/home/data/lj/Piano/saved/2_0_3_img','start_frame':104},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/3_1.txt','img_dir':'/home/data/lj/Piano/saved/3_1_img','start_frame':74},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/3_2.txt','img_dir':'/home/data/lj/Piano/saved/3_2_img','start_frame':78},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/3_3.txt','img_dir':'/home/data/lj/Piano/saved/3_3_img','start_frame':63},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/3_4.txt','img_dir':'/home/data/lj/Piano/saved/3_4_img','start_frame':69},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/4_1.txt','img_dir':'/home/data/lj/Piano/saved/4_1_img','start_frame':62},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/4_2.txt','img_dir':'/home/data/lj/Piano/saved/4_2_img','start_frame':63},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/4_3.txt','img_dir':'/home/data/lj/Piano/saved/4_3_img','start_frame':73},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/4_4.txt','img_dir':'/home/data/lj/Piano/saved/4_4_img','start_frame':60},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/5_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/5_+45_1_img','start_frame':63},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/5_0_1.txt','img_dir':'/home/data/lj/Piano/saved/5_0_1_img','start_frame':87},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/5_0_2.txt','img_dir':'/home/data/lj/Piano/saved/5_0_2_img','start_frame':69},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/6_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/6_+45_1_img','start_frame':54},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/6_0_1.txt','img_dir':'/home/data/lj/Piano/saved/6_0_1_img','start_frame':96},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/7_1.txt','img_dir':'/home/data/lj/Piano/saved/7_1_img','start_frame':80},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/7_2.txt','img_dir':'/home/data/lj/Piano/saved/7_2_img','start_frame':110},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/7_3.txt','img_dir':'/home/data/lj/Piano/saved/7_3_img','start_frame':52},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/7_4.txt','img_dir':'/home/data/lj/Piano/saved/7_4_img','start_frame':38},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/8_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/8_+45_1_img','start_frame':73},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/8_+45_2.txt','img_dir':'/home/data/lj/Piano/saved/8_+45_2_img','start_frame':72},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/8_0_1.txt','img_dir':'/home/data/lj/Piano/saved/8_0_1_img','start_frame':72},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/8_0_2.txt','img_dir':'/home/data/lj/Piano/saved/8_0_2_img','start_frame':108},
    
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/9_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/9_+45_1_img','start_frame':85},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/9_+45_2.txt','img_dir':'/home/data/lj/Piano/saved/9_+45_2_img','start_frame':67},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/9_0_1.txt','img_dir':'/home/data/lj/Piano/saved/9_0_1_img','start_frame':72},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/9_0_2.txt','img_dir':'/home/data/lj/Piano/saved/9_0_2_img','start_frame':60},
    
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/10_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/10_+45_1_img','start_frame':100},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/10_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/10_+45_2_img','start_frame':145},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/10_0_1.txt','img_dir':'/home/data/lj/Piano/saved/10_0_1_img','start_frame':72},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/10_0_2.txt','img_dir':'/home/data/lj/Piano/saved/10_0_2_img','start_frame':66},
    
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/11_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/11_+45_1_img','start_frame':89},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/11_+45_2.txt','img_dir':'/home/data/lj/Piano/saved/11_+45_2_img','start_frame':46},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/11_+45_3.txt','img_dir':'/home/data/lj/Piano/saved/11_+45_3_img','start_frame':77},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/11_0_1.txt','img_dir':'/home/data/lj/Piano/saved/11_0_1_img','start_frame':69},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/11_0_2.txt','img_dir':'/home/data/lj/Piano/saved/11_0_2_img','start_frame':49},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/11_0_3.txt','img_dir':'/home/data/lj/Piano/saved/11_0_3_img','start_frame':30},
    
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/12_1.txt','img_dir':'/home/data/lj/Piano/saved/12_1_img','start_frame':58},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/12_2.txt','img_dir':'/home/data/lj/Piano/saved/12_2_img','start_frame':68},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/12_3.txt','img_dir':'/home/data/lj/Piano/saved/12_3_img','start_frame':79},

    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/13_+45_1.txt','img_dir':'/home/data/lj/Piano/saved/13_+45_1_img','start_frame':73},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/13_+45_2.txt','img_dir':'/home/data/lj/Piano/saved/13_+45_2_img','start_frame':76},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/13_+45_3.txt','img_dir':'/home/data/lj/Piano/saved/13_+45_3_img','start_frame':68},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/13_0_1.txt','img_dir':'/home/data/lj/Piano/saved/13_0_1_img','start_frame':42},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/13_0_2.txt','img_dir':'/home/data/lj/Piano/saved/13_0_2_img','start_frame':47},
    
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/14_1.txt','img_dir':'/home/data/lj/Piano/saved/14_1_img','start_frame':52},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/14_2.txt','img_dir':'/home/data/lj/Piano/saved/14_2_img','start_frame':51},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/14_3.txt','img_dir':'/home/data/lj/Piano/saved/14_3_img','start_frame':77},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TrainSet/midi/10_0_2.txt','img_dir':'/home/data/lj/Piano/saved/10_0_2_img','start_frame':66}
]

VAL_ITEMS = [
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V1.txt','img_dir':'/home/data/lj/Piano/saved/V1','start_frame':98},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V2.txt','img_dir':'/home/data/lj/Piano/saved/V2','start_frame':72},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V3.txt','img_dir':'/home/data/lj/Piano/saved/V3','start_frame':51},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V4.txt','img_dir':'/home/data/lj/Piano/saved/V4','start_frame':75},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V5.txt','img_dir':'/home/data/lj/Piano/saved/V5','start_frame':110},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V6.txt','img_dir':'/home/data/lj/Piano/saved/V6','start_frame':101},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V7.txt','img_dir':'/home/data/lj/Piano/saved/V7','start_frame':87},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V8.txt','img_dir':'/home/data/lj/Piano/saved/V8','start_frame':94},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V9.txt','img_dir':'/home/data/lj/Piano/saved/V9','start_frame':122},
    {'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/midi/V10.txt','img_dir':'/home/data/lj/Piano/saved/V10','start_frame':85}
]

train_root = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/train'
val_root = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/val'

def load_key_dict():
    white_dict = {}
    black_dict = {} 
    black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                 31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                 60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
    white_num = [x for x in range(1, 89) if x not in black_num]
    for ix,item in enumerate(white_num):
        white_dict[item] = str(ix+1)

    for ix,item in enumerate(black_num):
        black_dict[item] = str(ix+1)
    
    return white_dict,black_dict 

def read_pitch_file(filename):
    with open(filename,'r') as fr:
        items = [l.strip() for l in fr.readlines()]
    pitches,onsets,offsets = [],[],[]
    for item in items:
        item = item.split(' ')
        pitches.append(item[0])
        onsets.append(item[1])
        offsets.append(item[2])
    return pitches,onsets,offsets 

def read_img_dir(img_dir):
    img_files = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if 'jpg' in x]
    img_files.sort()
    return img_files 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def new_save_dir(img_dir,mode):
    if img_dir[-1]=='/':
        img_dir = img_dir[:-1]
    root = train_root if mode=='train' else val_root
    save_white_pos_dir = os.path.join(root,'white','1',img_dir.split('/')[-1])
    save_white_neg_dir = os.path.join(root,'white','0',img_dir.split('/')[-1])
    save_black_pos_dir = os.path.join(root,'black','1',img_dir.split('/')[-1])
    save_black_neg_dir = os.path.join(root,'black','0',img_dir.split('/')[-1])
    exits = False 
    if os.path.exists(save_white_pos_dir):
        exits=True 
    ensure_dir(save_white_pos_dir)
    ensure_dir(save_white_neg_dir)
    ensure_dir(save_black_pos_dir)
    ensure_dir(save_black_neg_dir)
    return save_white_pos_dir,save_white_neg_dir,save_black_pos_dir,save_black_neg_dir,exits


def move_file(img_lists,save_dir,pos=True):
    for imgpath in img_lists:
        dst_p = os.path.join(save_dir,os.path.basename(imgpath))
        img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
        _,img = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
        count = len(np.where(img>0)[0])
        if np.random.random()>0.05 and count<100 and not pos:
            continue 
        dst_p = os.path.join(save_dir,os.path.basename(imgpath))
        shutil.copyfile(imgpath,dst_p)

def helper(pitch_file,img_dir,start_frame,mode='train'):
    pitches,onsets,offsets = read_pitch_file(pitch_file)
    white_dict,black_dict = load_key_dict()
    img_files = read_img_dir(img_dir)
    save_white_pos_dir,save_white_neg_dir,save_black_pos_dir,save_black_neg_dir,exists = new_save_dir(img_dir,mode)
    if exists:return 
    resoultion = 1.0/20
    
    press_white_imgs,press_black_imgs = [],[]
    
    white_path = os.path.join(img_dir,'press_white_img')
    black_path = os.path.join(img_dir,'press_black_img')

    total_white_imgs = [os.path.join(white_path,x) for x in os.listdir(white_path)]
    total_black_imgs = [os.path.join(black_path,x) for x in os.listdir(black_path)]
    for idx,onset in enumerate(onsets):
        offset = offsets[idx]
        pitch = pitches[idx]
        str_frame = round(float(onset)/resoultion + start_frame)
        end_frame = round(float(offset)/resoultion + start_frame)

        if pitch in white_dict.keys():
            index = white_dict[pitch]
            for frame in range(str_frame,end_frame):
                cur_frame = str(frame).zfill(4)
                filename = os.path.join(white_path,'{}_{}.jpg'.format(cur_frame,index))
                if os.path.exists(filename):
                    press_white_imgs.append(filename)
                    total_white_imgs.remove(filename)
        elif pitch in black_dict.keys():
            index = black_dict[pitch]
            for frame in range(str_frame,end_frame):
                cur_frame = str(frame).zfill(4)
                filename = os.path.join(black_path,'{}_{}.jpg'.format(cur_frame,index))
                if os.path.exists(filename):
                    press_black_imgs.append(filename)
                    total_black_imgs.remove(filename)
    total_white_imgs = np.array(total_white_imgs)
    total_black_imgs = np.array(total_black_imgs)

    move_file(press_white_imgs,save_white_pos_dir)
    move_file(total_white_imgs,save_white_neg_dir,False)
    move_file(press_black_imgs,save_black_pos_dir)
    move_file(total_black_imgs,save_black_neg_dir,False)

def main():
    for i,item in enumerate(TRAIN_ITEMS):
        pitch_file = item['midi']
        img_dir = item['img_dir']
        start_frame = item['start_frame']
        print(img_dir)
        helper(pitch_file,img_dir,start_frame)

    for i,item in enumerate(VAL_ITEMS):
        pitch_file = item['midi']
        img_dir = item['img_dir']
        start_frame = item['start_frame']
        print(img_dir)
        helper(pitch_file,img_dir,start_frame,mode='val')


if __name__=='__main__':
    main()
