import os 
import shutil 
import cv2
from PIL import Image 
from IPython import embed 
import numpy as np 
import mido 
from tqdm import tqdm
import argparse 

TRAIN_ITEMS = [
    {'midi':'/home/data/gxdata/Record/1120/midi/originBlack.txt','img_dir':'/home/data/lj/Piano/saved/originBlack','start_frame':156,'fps':24},
    {'midi':'/home/data/gxdata/Record/1120/midi/originWhite.txt','img_dir':'/home/data/lj/Piano/saved/originWhite','start_frame':245,'fps':24},
    {'midi':'/home/data/gxdata/Record/1120/midi/ExtraLightBlack.txt','img_dir':'/home/data/lj/Piano/saved/ExtraLightBlack','start_frame':97,'fps':24},
    {'midi':'/home/data/gxdata/Record/1120/midi/ExtraLightWhite.txt','img_dir':'/home/data/lj/Piano/saved/ExtraLightWhite','start_frame':142,'fps':24},

    {'midi':'/home/data/gxdata/Record/1121/random1.MID','img_dir':'/home/data/lj/Piano/saved/random1','start_frame':164,'fps':24},
    {'midi':'/home/data/gxdata/Record/1121/random4.MID','img_dir':'/home/data/lj/Piano/saved/random4','start_frame':173,'fps':24},
    {'midi':'/home/data/gxdata/Record/1121/extraLightBlack2.MID','img_dir':'/home/data/lj/Piano/saved/extraLightBlack2','start_frame':83,'fps':24},
    {'midi':'/home/data/gxdata/Record/1121/extraLightWhite2.MID','img_dir':'/home/data/lj/Piano/saved/extraLightWhite2','start_frame':105,'fps':24},
    {'midi':'/home/data/gxdata/Record/1121/x_lulu_-15.MID','img_dir':'/home/data/lj/Piano/saved/x_lulu_-15','start_frame':121,'fps':24},

    {'midi':'/home/data/gxdata/Record/1126/blackWhite.MID','img_dir':'/home/data/lj/Piano/saved/blackWhite','start_frame':52,'fps':25},
    {'midi':'/home/data/gxdata/Record/1126/Chord.MID','img_dir':'/home/data/lj/Piano/saved/Chord','start_frame':93,'fps':25},

    {'midi':'/home/data/gxdata/Record/1203/1-white.MID','img_dir':'/home/data/lj/Piano/saved/1-white','start_frame':157,'fps':25},
    {'midi':'/home/data/gxdata/Record/1203/2-black.MID','img_dir':'/home/data/lj/Piano/saved/2-black','start_frame':95,'fps':25},
    {'midi':'/home/data/gxdata/Record/1203/3-mix.MID','img_dir':'/home/data/lj/Piano/saved/3-mix','start_frame':161,'fps':25},
    {'midi':'/home/data/gxdata/Record/1203/5-mix.MID','img_dir':'/home/data/lj/Piano/saved/5-mix','start_frame':302,'fps':25},

    {'midi':'/home/data/gxdata/Record/1204/litM_camL_kb30.MID','img_dir':'/home/data/lj/Piano/saved/litM_camL_kb30','start_frame':54,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litM_camL_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litM_camL_kbM30','start_frame':51,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litM_camM_kb0.MID','img_dir':'/home/data/lj/Piano/saved/litM_camM_kb0','start_frame':95,'fps':25},

    {'midi':'/home/data/gxdata/Record/1204/litM_camM_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litM_camM_kbM30','start_frame':43,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litM_camR_kb0.MID','img_dir':'/home/data/lj/Piano/saved/litM_camR_kb0','start_frame':202,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litM_camR_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litM_camR_kbM30','start_frame':82,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litM_camL_kb0.MID','img_dir':'/home/data/lj/Piano/saved/litM_camL_kb0','start_frame':126,'fps':25},
    
    {'midi':'/home/data/gxdata/Record/1204/litL_camM_kb30.MID','img_dir':'/home/data/lj/Piano/saved/litL_camM_kb30','start_frame':42,'fps':24},
    {'midi':'/home/data/gxdata/Record/1204/litL_camL_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litL_camL_kbM30','start_frame':21,'fps':24},

    {'midi':'/home/data/gxdata/Record/1204/litLR_camM_kb0.MID','img_dir':'/home/data/lj/Piano/saved/litLR_camM_kb0','start_frame':55,'fps':24},
    {'midi':'/home/data/gxdata/Record/1204/litLR_camM_kb30.MID','img_dir':'/home/data/lj/Piano/saved/litLR_camM_kb30','start_frame':54,'fps':24},

    {'midi':'/home/data/gxdata/Record/1204/litLR_camM_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litLR_camM_kbM30','start_frame':89,'fps':24},
    {'midi':'/home/data/gxdata/Record/1204/litR_camM_kb0.MID','img_dir':'/home/data/lj/Piano/saved/litR_camM_kb0','start_frame':51,'fps':24},
    {'midi':'/home/data/gxdata/Record/1204/litR_camM_kb30.MID','img_dir':'/home/data/lj/Piano/saved/litR_camM_kb30','start_frame':36,'fps':24},

    {'midi':'/home/data/gxdata/Record/1121/random2.MID','img_dir':'/home/data/lj/Piano/saved/random2','start_frame':174,'fps':24},
    {'midi':'/home/data/gxdata/Record/1121/random3.MID','img_dir':'/home/data/lj/Piano/saved/random3','start_frame':161,'fps':24},
    {'midi':'/home/data/gxdata/Record/1203/4-mix.MID','img_dir':'/home/data/lj/Piano/saved/4-mix','start_frame':126,'fps':25},

    {'midi':'/home/data/gxdata/Record/1204/litM_camM_kb30.MID','img_dir':'/home/data/lj/Piano/saved/litM_camM_kb30','start_frame':83,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litM_camR_kb30.MID','img_dir':'/home/data/lj/Piano/saved/litM_camR_kb30','start_frame':146,'fps':25},
    {'midi':'/home/data/gxdata/Record/1204/litL_camM_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litL_camM_kbM30','start_frame':85,'fps':24},
    {'midi':'/home/data/gxdata/Record/1204/litR_camM_kbM30.MID','img_dir':'/home/data/lj/Piano/saved/litR_camM_kbM30','start_frame':50,'fps':24}
]
VAL_ITEMS = [
    {'midi':'/home/data/gxdata/Record/1223_2/140.MID','img_dir':'/home/data/lj/Piano/saved/middle_140','start_frame':201,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/280.MID','img_dir':'/home/data/lj/Piano/saved/middle_280','start_frame':164,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/400.MID','img_dir':'/home/data/lj/Piano/saved/middle_400','start_frame':79,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/510.MID','img_dir':'/home/data/lj/Piano/saved/middle_510','start_frame':173,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/600.MID','img_dir':'/home/data/lj/Piano/saved/middle_600','start_frame':161,'fps':24},

    {'midi':'/home/data/gxdata/Record/1223_2/140.MID','img_dir':'/home/data/lj/Piano/saved/left_140','start_frame':264,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/280.MID','img_dir':'/home/data/lj/Piano/saved/left_280','start_frame':210,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/400.MID','img_dir':'/home/data/lj/Piano/saved/left_400','start_frame':34,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/510.MID','img_dir':'/home/data/lj/Piano/saved/left_510','start_frame':125,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/600.MID','img_dir':'/home/data/lj/Piano/saved/left_600','start_frame':103,'fps':24},

    {'midi':'/home/data/gxdata/Record/1223_2/140.MID','img_dir':'/home/data/lj/Piano/saved/right_140','start_frame':112,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/280.MID','img_dir':'/home/data/lj/Piano/saved/right_280','start_frame':95,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/400.MID','img_dir':'/home/data/lj/Piano/saved/right_400','start_frame':127,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/510.MID','img_dir':'/home/data/lj/Piano/saved/right_510','start_frame':240,'fps':24},
    {'midi':'/home/data/gxdata/Record/1223_2/600.MID','img_dir':'/home/data/lj/Piano/saved/right_600','start_frame':230,'fps':24},
    {'midi':'/home/data/gxdata/Record/1225/2_baseline.MID','img_dir':'/home/data/lj/Piano/saved/2_baseline','start_frame':94,'fps':24}
]


train_root = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/train'
val_root = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/val'

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

def processMidi(midiPath,midi_offset=1.5):
    mid = mido.MidiFile(midiPath)
    timeCount = 0
    dataList = []
    for msg in mid:
        if not msg.is_meta:
            if msg.type == 'control_change':
                timeCount = timeCount + msg.time
            elif msg.type == 'note_on' or msg.type == 'note_off':
                timeCount = timeCount + msg.time
                if 'wmv' in midiPath:
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
    pitch_onset_offset = []
    for item in result:
        po = [item[2] - midi_offset, item[3] - midi_offset,item[0]]
        pitch_onset_offset.append(po)
    pitch_onset_offset = sorted(pitch_onset_offset,key=lambda x:(x[0],x[1],x[2]))
    pitches,onsets,offsets = [],[],[]
    for idx,pof in enumerate(pitch_onset_offset):
        pitches.append(pof[2])
        onsets.append(pof[0])
        offsets.append(pof[1])
    return pitches,onsets,offsets

def new_save_dir(img_dir,mode='train'):
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
    for imgpath in tqdm(img_lists):
        img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
        _,img = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
        count = len(np.where(img>0)[0])
        if np.random.random()>0.05 and count<100 and not pos:
            continue 
        dst_p = os.path.join(save_dir,os.path.basename(imgpath))
        shutil.copyfile(imgpath,dst_p)

def helper(pitch_file,img_dir,start_frame,fps,mode='train'):
    if 'txt' in pitch_file:
        pitches,onsets,offsets = read_pitch_file(pitch_file)
    else:
        pitches,onsets,offsets = processMidi(pitch_file)
    white_dict,black_dict = load_key_dict()
    save_white_pos_dir,save_white_neg_dir,save_black_pos_dir,save_black_neg_dir,exits = new_save_dir(img_dir,mode)
    if exits:return 
    resoultion = 1.0/fps

    press_white_imgs,press_black_imgs = [],[]
    
    white_path = os.path.join(img_dir,'press_white_img')
    black_path = os.path.join(img_dir,'press_black_img')

    total_white_imgs = [os.path.join(white_path,x) for x in os.listdir(white_path)]
    total_black_imgs = [os.path.join(black_path,x) for x in os.listdir(black_path)]
    for idx,onset in enumerate(onsets):
        offset = offsets[idx]
        pitch = str(pitches[idx])
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
        fps = item['fps']
        print(img_dir)
        helper(pitch_file,img_dir,start_frame,fps)
    
    for i,item in enumerate(VAL_ITEMS):
        pitch_file = item['midi']
        img_dir = item['img_dir']
        start_frame = item['start_frame']
        fps = item['fps']
        print(img_dir)
        helper(pitch_file,img_dir,start_frame,fps,'val')

if __name__=='__main__':
    main()
