import os
import cv2
from PIL import Image
import numpy as np
import argparse 
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('--video',type=str,default=None)
parser.add_argument('--root',type=str,default=None)
args = parser.parse_args()

def img2video(path,size=(1280,720)):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vw = cv2.VideoWriter('output.mp4', fourcc=fourcc, fps=25.0, frameSize=size)
    img_files = [os.path.join(path,x) for x in os.listdir(path)]
    img_files.sort()
    for file in img_files:
        f_read = cv2.imread(file)
        f_img = Image.fromarray(f_read)  
        f_rs = f_img.resize(list(size),resample=Image.NONE)
        f_out = np.array(f_rs)
        vw.write(f_out)
    vw.release()



def readvideo2flipvideo(video_path,save_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError('read video wrong')
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(fps)
    #return 
    print(video_path)
    save_img_dir = os.path.join(os.path.split(save_path)[0],'images',os.path.basename(save_path).split('.')[0])
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    else:return 
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    videowriter = cv2.VideoWriter(save_path,fourcc=fourcc,fps=fps,frameSize=size)
    count = 0
    for idx in tqdm(range(total_frames)):
        ret,frame = capture.read()
        if not ret:
            break 
        frame = cv2.flip(frame,-1)
        f_img = Image.fromarray(frame)
        f_re = f_img.resize(list(size),resample=Image.NONE)
        f_out = np.array(f_re)
        videowriter.write(f_out) 
        save_img = os.path.join(save_img_dir,'{}.jpg'.format(str(count).zfill(4)))
        cv2.imwrite(save_img,f_out)
        count+=1
    videowriter.release()



def list_wmvdir(path,wmv_files):
    subpaths = [os.path.join(path,x) for x in os.listdir(path)]
    for spath in subpaths:
        if os.path.isdir(spath):
            list_wmvdir(spath,wmv_files)
        elif spath.endswith('wmv') or spath.endswith('MP4'):
            wmv_files.append(spath)


def main(wmv_files):
    if isinstance(wmv_files,list):
        wmv_files.sort()
        for video_path in wmv_files:
            if 'wmv' in video_path:
                save_path = video_path.replace('wmv','mp4')
            elif 'MP4' in video_path:
                save_path = video_path.replace('MP4','mp4')
            readvideo2flipvideo(video_path,save_path)
    else:
        save_path = wmv_files.replace('MP4','mp4')
        print(save_path)
        readvideo2flipvideo(wmv_files,save_path)
        

if __name__=='__main__':
    if args.root is not None: 
        train_files = []
        list_wmvdir(args.root,train_files)
        main(train_files)
    if args.video is not None:
        main(args.video)
