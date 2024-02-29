import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
from models import PSPNet
import cv2 
import time 

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def save_images(image, mask, output_path, image_file, palette):
	# Saves the image, the model output and the results after the post processing
    image_file = os.path.basename(image_file).split('.')[0]
    colorized_mask = colorize_mask(mask, palette)
    pmask = np.array(colorized_mask)
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    contours, hier = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cidx,cnt in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(cnt)
        if h>50:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imwrite(os.path.join(output_path,image_file+'.jpg'),image)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()

    # Dataset used for training the model
    MEAN = [0.45734706, 0.43338275, 0.40058118]
    STD = [0.23965294, 0.23532275, 0.2398498]

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(MEAN,STD)
    num_classes = 2
    palette = [0,0,0,128,0,128]

    # Model
    model = PSPNet(num_classes=num_classes,backbone='resnet18')
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            image = Image.open(img_file).convert('RGB')
            image = image.resize((480, 320))
            input = normalize(to_tensor(image)).unsqueeze(0)
            print(input.size())
            t1 = time.time()
            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            print(time.time()-t1)
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            save_images(image, prediction, args.output, img_file, palette)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
