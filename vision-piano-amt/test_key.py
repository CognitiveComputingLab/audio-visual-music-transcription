import os 
import argparse 
from models.model_helper import ModelProduct 
import argparse 
from PIL import Image 
import sys 
from config import PROJECT_ROOT,cfg
sys.path.append(os.path.join(PROJECT_ROOT,'piano_utils'))
from keyboard import KeyBoard
import numpy as np 
import cv2 
from tqdm import tqdm 

def str2bool(param):
    return param in ['True','1','true','yes']

parser = argparse.ArgumentParser()
parser.add_argument('--img','-i',type=str,default=None)
parser.add_argument('--type',default='white',choices=['white','black'])
parser.add_argument('--style',default='key',choices=['keyboard','key'])
parser.add_argument('--img_dir','-id',type=str,default=None)
parser.add_argument('--pos',type=str,default='True')
parser.add_argument('--model_dir',type=str,default=None)
args = parser.parse_args()

save_dir = '/home/data/lj/Piano/saved/other'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def list_models(path):
    model_files = [os.path.join(path,x) for x in os.listdir(path) if x.endswith('pth') ]
    model_files.sort()
    return model_files

def list_img_dir(path,img_lists):
    subpaths = [os.path.join(path,x) for x in os.listdir(path)]
    for subpath in subpaths:
        if os.path.isdir(subpath):
            list_img_dir(subpath,img_lists)
        elif subpath.endswith('jpg'):
            img_lists.append(subpath)

def list_img_dirs(path,img_dirs):
    subpaths = [os.path.join(path,x) for x in os.listdir(path)]
    for subpath in subpaths:
        if os.path.isdir(subpath):
            img_dirs.append(subpath)

if args.img is not None:
    if args.style=='key':
        modelproduct = ModelProduct()
        input_imgs = []
        img = Image.open(args.img)
        input_imgs.append(img)
        if args.type=='black':
            pred = modelproduct.detect_black_keys(input_imgs,debug=True)
        else: 
            pred = modelproduct.detect_white_keys(input_imgs,debug=True)
        print(pred)
    else:
        keyboard = KeyBoard()
        img = Image.open(args.img)
        keyboard.detect_keyboard(img)

if args.img_dir is not None:
    if args.model_dir is not None:
        img_lists = []
        list_img_dir(args.img_dir,img_lists)
        model_files = list_models(args.model_dir)
        for model_file in model_files:
            if args.type=='black':
                modelproduct = ModelProduct(black_model_file=model_file)
            else:
                modelproduct = ModelProduct(white_model_file=model_file)
            count = 0
            for img_path in tqdm(img_lists):
                input_imgs = []
                img = Image.open(img_path)
                input_imgs.append(img)
                if args.type=='black':
                    pred,prob = modelproduct.detect_black_keys(input_imgs,debug=False)
                else:
                    pred,prob = modelproduct.detect_white_keys(input_imgs,debug=False)
                if str2bool(args.pos):
                    if pred[0]==1:count+=1
                    else:
                        split_line = img_path.split('/')
                        save_file = os.path.join(save_dir,'{}_{}'.format(split_line[-2],split_line[-1]))
                        img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_file,img)
                else:
                    if pred[0]==0:count+=1
            print('{} {}/{}={:.3}'.format(model_file,count,len(img_lists),count/len(img_lists)))
    else:    
        modelproduct = ModelProduct()
        img_dirs = []
        list_img_dirs(args.img_dir,img_dirs)
        if len(img_dirs)==0:
            img_dirs = [args.img_dir]
        for img_dir in img_dirs:
            count = 0
            img_lists = []
            list_img_dir(img_dir,img_lists)
            for img_path in tqdm(img_lists):
                input_imgs = []
                img = Image.open(img_path)
                input_imgs.append(img)
                if args.type=='black':
                    pred,prob = modelproduct.detect_black_keys(input_imgs,debug=False)
                else:
                    pred,prob = modelproduct.detect_white_keys(input_imgs,debug=False)
                if str2bool(args.pos):
                    if pred[0]==1:count+=1
                    else:
                        split_line = img_path.split('/')
                        save_file = os.path.join(save_dir,'{}_{}'.format(split_line[-2],split_line[-1]))
                        img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_file,img)
                else:
                    if pred[0]==0:count+=1
            if len(img_lists)==0:
                print(img_dir)
            else:
                print('{} {}/{}={:.3}'.format(img_dir,count,len(img_lists),count/len(img_lists)))
        


