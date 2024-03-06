import os 
from easydict import EasyDict 

_C = EasyDict()
cfg = _C 
_C.Paper_WHITE_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/white_train.npy.npz'
_C.Paper_WHITE_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/white_val.npy.npz'
_C.Paper_BLACK_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/black_train.npy.npz'
_C.Paper_BLACK_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/paperData1/black_val.npy.npz'


_C.Own_WHITE_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/white_train_add_network.npy.npz'
_C.Own_WHITE_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/white_val.npy.npz'
_C.Own_BLACK_TRAIN_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/black_train_add_network.npy.npz'
_C.Own_BLACK_VAL_FILE = '/home/data/lj/Piano/KEY_PRESS/data/owndataset/black_val.npy.npz'

_C.SAVE_MODEL_DIR = '/home/data/lj/Piano/KEY_PRESS/checkpoint/experiment'


_C.INPUT_SIZE = {
        'white':{
            'paperdata':[112,32],
            'owndata':[112,32]
        },
        'black':{
            'paperdata':[112,32],
            'owndata':[112,32]
        }
}
       
_C.ALPHA = {
        'white':{
            'paperdata':3.0,
            'owndata':3.0
        },
        'black':{
            'paperdata':2.0,
            'owndata':2.0
        }
}

