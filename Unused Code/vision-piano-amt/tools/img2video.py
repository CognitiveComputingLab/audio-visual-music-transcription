import os
import cv2
from PIL import Image
import numpy as np
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',type=str,default=None)
parser.add_argument('--img_root',type=str,default=None)
parser.add_argument('--type',default='diff',type=str,choices=['diff','total'])

args = parser.parse_args()


def img2video(img_dir,video_file):
    imgs = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('jpg')]
    assert len(imgs)>0,print('imgs error')
    img = cv2.imread(imgs[0])
    h,w,_ = img.shape 
    size = (w,h)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(video_file,fourcc=fourcc, fps=25.0, frameSize=size)
    imgs.sort()
    for img_file in tqdm(imgs):
        img = cv2.imread(img_file)
        fimg = Image.fromarray(img)  
        fimg = fimg.resize(list(size),resample=Image.NONE)
        img = np.array(fimg)
        videoWriter.write(img)
    videoWriter.release()
        


def list_all_img_dirs():
    subpaths = [os.path.join(args.img_root,x) for x in os.listdir(args.img_root)]
    if args.type=='diff':
        all_img_dirs = [os.path.join(x,'diff_img') for x in subpaths]
    else:
        all_img_dirs = [os.path.join(x,'detect_total_img') for x in subpaths]
    return all_img_dirs 

if __name__=='__main__':
    if args.img_dir is not None:
        img_dir = args.img_dir 
        if img_dir[-1]=='/':img_dir = img_dir[0:-1]
        if args.type=='total':
            video_file = os.path.join(os.path.split(img_dir)[0],img_dir.split('/')[-2]+'.mp4')
        else:
            video_file = os.path.join(os.path.split(img_dir)[0],img_dir.split('/')[-2]+'_diff.mp4')
        if os.path.exists(video_file):
            os.remove(video_file)
        img2video(img_dir,video_file)
    if args.img_root is not None:
        all_img_dirs = list_all_img_dirs()
        for img_dir in all_img_dirs:
            if img_dir[-1]=='/':img_dir = img_dir[0:-1]
            if args.type=='total':
                video_file = os.path.join(os.path.split(img_dir)[0],img_dir.split('/')[-2]+'.mp4')
            else:
                video_file = os.path.join(os.path.split(img_dir)[0],img_dir.split('/')[-2]+'_diff.mp4')
            print(video_file)
            if os.path.exists(video_file):
                os.remove(video_file)
            img2video(img_dir,video_file) 
