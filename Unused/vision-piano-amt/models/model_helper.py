import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import torchvision.transforms as transforms
import torch.nn.functional as F 
from torch.autograd import Variable
from PIL import Image
import cv2
import sys 
import os 
import numpy as np 
import time 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0,PROJECT_ROOT)
from config import cfg 
from .hand_model import build_s3fd 
from .simple import SimpleNet
from .resnet_112_32 import ResNet18 as ResNet18_112 
from .conv3net import Conv3Net
from IPython import embed 

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


class ModelProduct(object):
    def __init__(self,white_model_file=None,black_model_file=None):
        self.load_det_hand_model() 
        self.load_white_key_model(white_model_file)
        self.load_black_key_model(black_model_file)
        #print('->>finish det hand model load')
        #print('->>finish whitekey model load')
        #print('->>finish blackkey model load')

        if cfg.WHITE_INPUT_CHANNEL==3:
            self.transform_white = transforms.Compose([
                transforms.Resize(cfg.WHITE_INPUT_SIZE),
                transforms.ToTensor(),
            ])
        else:
            self.transform_white = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(cfg.WHITE_INPUT_SIZE),
                transforms.ToTensor(),
            ])
        if cfg.BLACK_INPUT_CHANNEL==3:
            self.transform_black = transforms.Compose([
                transforms.Resize(cfg.BLACK_INPUT_SIZE),
                transforms.ToTensor(),
            ])
        else:
            self.transform_black = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(cfg.BLACK_INPUT_SIZE),
                transforms.ToTensor(),
            ])
        
    def load_det_hand_model(self):
        self.hand_net = build_s3fd('test',cfg.NUM_CLASSES)
        self.hand_net.load_state_dict(torch.load(cfg.HAND_MODEL))
        self.hand_net.eval()
        if torch.cuda.is_available():
            self.hand_net.cuda()
        cudnn.benchmark = True 

    def load_white_key_model(self,white_model_file): 
        if cfg.WNetStyle=='resnet':
            self.white_key_net = ResNet18_112(input_channel=cfg.WHITE_INPUT_CHANNEL,
                                        base_channel=16,
                                        num_classes=cfg.KEY_NUM_CLASSES)
        elif cfg.WNetStyle =='simple':
            self.white_key_net = SimpleNet(input_channel=cfg.WHITE_INPUT_CHANNEL,
                                        num_classes=cfg.KEY_NUM_CLASSES)
        else:
            self.white_key_net = Conv3Net(input_channel=cfg.WHITE_INPUT_CHANNEL,
                                        num_classes=cfg.KEY_NUM_CLASSES)
        if white_model_file is None:
            self.white_key_net.load_state_dict(torch.load(cfg.KEY_WHITE_MODEL))
        else:
            self.white_key_net.load_state_dict(torch.load(white_model_file))
        self.white_key_net.eval()
        if torch.cuda.is_available():
            self.white_key_net.cuda()

    def load_black_key_model(self,black_model_file): 
        if cfg.BNetStyle=='resnet':
            self.black_key_net = ResNet18_112(input_channel = cfg.BLACK_INPUT_CHANNEL,
                                         base_channel=16,
                                         num_classes = cfg.KEY_NUM_CLASSES)
        elif cfg.BNetStyle=='simple':
            self.black_key_net = SimpleNet(input_channel = cfg.BLACK_INPUT_CHANNEL,
                                         num_classes = cfg.KEY_NUM_CLASSES,type='black')
        else:
             self.black_key_net = Conv3Net(input_channel = cfg.BLACK_INPUT_CHANNEL,
                                         num_classes = cfg.KEY_NUM_CLASSES,type='black')
        if black_model_file is None:
            self.black_key_net.load_state_dict(torch.load(cfg.KEY_BLACK_MODEL))
        else:
            self.black_key_net.load_state_dict(torch.load(black_model_file))
        self.black_key_net.eval()
        if torch.cuda.is_available():
            self.black_key_net.cuda()

    def detect_white_keys(self,imgs,debug=False):
        inputs = list()
        t1 = time.time()
        with torch.no_grad():
            for img in imgs:
                img = self.transform_white(img).unsqueeze(0)
                img = img.cuda()
                inputs.append(img)
            inputs = torch.cat(inputs,dim=0)
            #print('preprocess cost {}'.format(time.time()-t1))
            t1 = time.time()
            output = self.white_key_net(inputs)
            prob = F.softmax(output, dim=1)   #----按行softmax,行的和概率为1,每个元素代表着概
            prob = prob.cpu().numpy()
            #print('detect cost {}'.format(time.time()-t1))
            result = (prob[:,1]>cfg.WHITE_KEY_THRESH).astype(int)
            pred = np.where(prob[:,1]>cfg.WHITE_KEY_THRESH)[0]
            if debug:
                embed()
            for i in range(1,len(pred)):
                if (pred[i]-pred[i-1])==1:
                    if prob[pred[i-1],1]<cfg.NEAR_KEY_THRESH and prob[pred[i],1]<cfg.NEAR_KEY_THRESH:
                        if prob[pred[i-1],1]<prob[pred[i],1]:
                            result[pred[i-1]] = 0 
                        else:
                            result[pred[i]] = 0 
                    else:
                        result[pred[i-1]:pred[i]+1] = (prob[pred[i-1]:pred[i]+1,1]>cfg.NEAR_KEY_THRESH).astype(int)
        return result,prob[:,1]
    
    def detect_black_keys(self,imgs,debug=False):
        inputs = list()
        with torch.no_grad():
            for img in imgs:
                img = self.transform_black(img).unsqueeze(0)
                img = img.cuda()
                inputs.append(img)
            inputs = torch.cat(inputs,dim=0)
            output = self.black_key_net(inputs)
            prob = F.softmax(output,dim=1)
            prob = prob.cpu().numpy()
            result = (prob[:,1]>cfg.BLACK_KEY_THRESH).astype(int)
            if debug:
                embed()
        return result,prob[:,1]

    def detect_hand(self,img,Rect):
        if img.mode == 'L':
            img = img.convert('RGB')
        img = np.array(img)
        height,width,_ = img.shape 
        img = img[Rect[1]:height,:]
        ori_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        max_im_shrink = np.sqrt(720 * 640 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink,fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
        x = to_chw_bgr(image).astype('float32')
        img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
        x -= img_mean
        x = x[[2, 1, 0], :, :]
        
        x = Variable(torch.from_numpy(x).unsqueeze(0))
        x = x.cuda()
        y = self.hand_net(x)
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        hand_box = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= cfg.VIS_THRESH:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:5] * scale).cpu().numpy()
                lr_mark = detections[0,i,j,5:].cpu().numpy()
                left_up, right_bottom = (int(pt[0]), int(pt[1]+Rect[1])), (int(pt[2]), int(pt[3]+Rect[1]))
                hand_box.append((left_up,right_bottom))
                j += 1
        return hand_box
        
