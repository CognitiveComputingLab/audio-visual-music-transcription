import os 
import sys 
from easydict import EasyDict 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

EXPERMENT_CONFIG = {
        'on':False,
        'alpha':False,
        'network':True
}

netstyle = 'conv3net'

_C = EasyDict()
cfg = _C 
# detect hand anchor config 
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# detect hand config 
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05
_C.NUM_CLASSES = 5 # Whether there is a hand in view
_C.VIS_THRESH = 0.7

# pretained model config
_C.HAND_MODEL = os.path.join(PROJECT_ROOT,'weights','det_hand.pth') # --> hand detection (PSPNet for hand; EgoHands and EgoYouTubeHands)
_C.KEY_WHITE_MODEL = './weights/experments/network/record/white_conv3net_add_network.pth' # --> white key detection (conv3net; piano video data with prominent white keys)
_C.KEY_BLACK_MODEL = './weights/experments/network/record/black_conv3net_add_network.pth' # --> black key detection (conv3net; piano video data with prominent black keys)
_C.MEAN = [0.45734706, 0.43338275, 0.40058118]
_C.STD = [0.23965294, 0.23532275, 0.2398498]

## key_config 
_C.KEY_NUM_CLASSES = 2 # Whether the key is pressed or not
_C.WHITE_KEY_THRESH = 0.6
_C.BLACK_KEY_THRESH = 0.4
_C.NEAR_KEY_THRESH = 0.95
_C.WHITE_INPUT_CHANNEL = 1
_C.BLACK_INPUT_CHANNEL = 1
_C.WHITE_INPUT_SIZE = [112,32]
_C.BLACK_INPUT_SIZE = [112,32]

##keyboard config 
_C.KEYBOARD_MODEL = os.path.join(PROJECT_ROOT,'weights','keyboard.pth') # --> keyboard detection (PSPNet for keyboard; keyboard images annotated with four corners with labelMe with diverse visual conditions)
_C.KEYBOARD_PALETTE = [0,0,0,64,0,128]
_C.KEYBOARD_NUM_CLASSES = 2 # Whether there is a keyboard or not

## save file config 
if not EXPERMENT_CONFIG['on']:
    _C.SAVE_IMG_DIR = os.path.join('/home/data/lj/Piano','saved','network')
else:
    if EXPERMENT_CONFIG['alpha']:
        _C.SAVE_IMG_DIR = os.path.join('/home/data/lj/Piano','saved','experment','alpha','alpha3.0')
    elif EXPERMENT_CONFIG['network']:
        _C.SAVE_IMG_DIR = os.path.join('/home/data/lj/Piano','saved','experment','network',netstyle)

## ['resnet','simple','conv3net']
_C.WNetStyle = netstyle
_C.BNetStyle = 'conv3net'

## hand segment 
_C.HAND_SEG_MODEL = os.path.join(PROJECT_ROOT,'weights','seg_hand.pth') # --> hand segmentation ()
_C.HAND_SEG_PALETTE = [0,0,0,64,0,128]
_C.HAND_SEG_NUM_CLASSES = 2 # Whether a hand is in view or not

### vision 
_C.VISION_DETECT = True 
_C.UPDATE_BACKGROUND = True 

_C.HAND_LENGTH = 25
###evalute
_C.EVALUATE_MAP = {
        'middle_140':{'start_frame':201,'midi':'/home/data/gxdata/Record/1223_2/140.MID','fps':24,'midi_offset':1.5}, 
        'middle_280':{'start_frame':164,'midi':'/home/data/gxdata/Record/1223_2/280.MID','fps':24,'midi_offset':1.5}, 
        'middle_400':{'start_frame':79,'midi':'/home/data/gxdata/Record/1223_2/400.MID','fps':24,'midi_offset':1.5}, 
        'middle_510':{'start_frame':173,'midi':'/home/data/gxdata/Record/1223_2/510.MID','fps':24,'midi_offset':1.5}, 
        'middle_600':{'start_frame':161,'midi':'/home/data/gxdata/Record/1223_2/600.MID','fps':24,'midi_offset':1.5}, 

        'left_140':{'start_frame':264,'midi':'/home/data/gxdata/Record/1223_2/140.MID','fps':24,'midi_offset':1.5}, 
        'left_280':{'start_frame':210,'midi':'/home/data/gxdata/Record/1223_2/280.MID','fps':24,'midi_offset':1.5}, 
        'left_400':{'start_frame':34,'midi':'/home/data/gxdata/Record/1223_2/400.MID','fps':24,'midi_offset':1.5}, 
        'left_510':{'start_frame':125,'midi':'/home/data/gxdata/Record/1223_2/510.MID','fps':24,'midi_offset':1.5}, 
        'left_600':{'start_frame':103,'midi':'/home/data/gxdata/Record/1223_2/600.MID','fps':24,'midi_offset':1.5}, 

        'right_140':{'start_frame':112,'midi':'/home/data/gxdata/Record/1223_2/140.MID','fps':24,'midi_offset':1.5}, 
        'right_280':{'start_frame':95,'midi':'/home/data/gxdata/Record/1223_2/280.MID','fps':24,'midi_offset':1.5}, 
        'right_400':{'start_frame':127,'midi':'/home/data/gxdata/Record/1223_2/400.MID','fps':24,'midi_offset':1.5}, 
        'right_510':{'start_frame':240,'midi':'/home/data/gxdata/Record/1223_2/510.MID','fps':24,'midi_offset':1.5}, 
        'right_600':{'start_frame':230,'midi':'/home/data/gxdata/Record/1223_2/600.MID','fps':24,'midi_offset':1.5}, 

        '1_baseline':{'start_frame':140,'midi':'/home/data/gxdata/Record/1225/1_baseline.MID','fps':24,'midi_offset':1.5}, 
        '1_right_260':{'start_frame':66,'midi':'/home/data/gxdata/Record/1225/1_right_260.MID','fps':24,'midi_offset':1.5}, 
        '1_right_400':{'start_frame':84,'midi':'/home/data/gxdata/Record/1225/1_right_400.MID','fps':24,'midi_offset':1.5}, 
        '1_right_520':{'start_frame':168,'midi':'/home/data/gxdata/Record/1225/1_right_520.MID','fps':24,'midi_offset':1.5}, 
        '1_right_630':{'start_frame':72,'midi':'/home/data/gxdata/Record/1225/1_right_630.MID','fps':24,'midi_offset':1.5}, 
        '1_right_730':{'start_frame':59,'midi':'/home/data/gxdata/Record/1225/1_right_730.MID','fps':24,'midi_offset':1.5},

        '2_baseline':{'start_frame':94,'midi':'/home/data/gxdata/Record/1225/2_baseline.MID','fps':24,'midi_offset':1.5}, 
        '2_right_280':{'start_frame':67,'midi':'/home/data/gxdata/Record/1225/2_right_280.MID','fps':24,'midi_offset':1.5},
        '2_right_400':{'start_frame':58,'midi':'/home/data/gxdata/Record/1225/2_right_420.MID','fps':24,'midi_offset':1.5}, 
        '2_right_520':{'start_frame':55,'midi':'/home/data/gxdata/Record/1225/2_right_520.MID','fps':24,'midi_offset':1.5}, 
        '2_right_630':{'start_frame':65,'midi':'/home/data/gxdata/Record/1225/2_right_630.MID','fps':24,'midi_offset':1.5}, 
        '2_right_730':{'start_frame':49,'midi':'/home/data/gxdata/Record/1225/2_right_730.MID','fps':24,'midi_offset':1.5}, 

        '3_middle_260':{'start_frame':121,'midi':'/home/data/gxdata/Record/1225/3_middle_260.MID','fps':24,'midi_offset':1.5}, 
        '3_middle_400':{'start_frame':51,'midi':'/home/data/gxdata/Record/1225/3_middle_400.MID','fps':24,'midi_offset':1.5},
        '3_middle_530':{'start_frame':46,'midi':'/home/data/gxdata/Record/1225/3_middle_530.MID','fps':24,'midi_offset':1.5}, 
        '3_middle_690':{'start_frame':70,'midi':'/home/data/gxdata/Record/1225/3_middle_690.MID','fps':24,'midi_offset':1.5}, 
        '3_middle_800':{'start_frame':55,'midi':'/home/data/gxdata/Record/1225/3_middle_800.MID','fps':24,'midi_offset':1.5},

        '4_left_240':{'start_frame':99,'midi':'/home/data/gxdata/Record/1225/4_left_240.MID','fps':24,'midi_offset':1.5}, 
        '4_left_390':{'start_frame':85,'midi':'/home/data/gxdata/Record/1225/4_left_390.MID','fps':24,'midi_offset':1.5}, 
        '4_left_520':{'start_frame':68,'midi':'/home/data/gxdata/Record/1225/4_left_520.MID','fps':24,'midi_offset':1.5}, 
        '4_left_620':{'start_frame':28,'midi':'/home/data/gxdata/Record/1225/4_left_620.MID','fps':24,'midi_offset':1.5}, 
        '4_left_730':{'start_frame':55,'midi':'/home/data/gxdata/Record/1225/4_left_730.MID','fps':24,'midi_offset':1.5}, 


        'level_1_no_2':{'start_frame':0,'midi':'/home/data/gxdata/netTanQinBa/netDataNew/videomidi/level_1_no_2.txt','fps':25,'midi_offset':0}, 


        'V1':{'start_frame':98,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V1.wmv.mid','fps':20,'midi_offset':0}, 
        'V2':{'start_frame':72,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V2.wmv.mid','fps':20,'midi_offset':0}, 
        'V3':{'start_frame':51,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V3.wmv.mid','fps':20,'midi_offset':0}, 
        'V4':{'start_frame':75,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V4.wmv.mid','fps':20,'midi_offset':0}, 
        'V5':{'start_frame':110,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V5.wmv.mid','fps':20,'midi_offset':0}, 
        'V6':{'start_frame':101,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V6.wmv.mid','fps':20,'midi_offset':0}, 
        'V7':{'start_frame':87,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V7.wmv.mid','fps':20,'midi_offset':0}, 
        'V8':{'start_frame':94,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V8.wmv.mid','fps':20,'midi_offset':0}, 
        'V9':{'start_frame':122,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V9.wmv.mid','fps':20,'midi_offset':0}, 
        'V10':{'start_frame':85,'midi':'/home/data/lj/Piano/paperData/IWSSIP/TestSet/OriginalVideos/V10.wmv.mid','fps':20,'midi_offset':0}, 
}

