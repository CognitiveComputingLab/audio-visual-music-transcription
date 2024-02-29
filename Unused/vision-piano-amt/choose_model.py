import os 
import sys
import argparse 
from visAmthelper import VisAmtHelper 
from config import cfg 

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',type=str,default=None)
parser.add_argument('--model_dir',type=str,default=None)
parser.add_argument('--type',type=str,default='white',choices=['white','black'])
parser.add_argument('--music_type',type=str,default='record',choices=['record','paper'])
args = parser.parse_args()

def list_models(path):
    model_files = [os.path.join(path,x) for x in os.listdir(path) if 'epoch' in x ]
    model_files.sort()
    return model_files

if __name__=='__main__': 
    if args.img_dir is not None:
        if args.img_dir[-1]=='/':
            args.img_dir = args.img_dir[0:-1]
        file_mark = args.img_dir.split('/')[-1]
        midi,start_frame,fps,midi_offset = None,0,25,0
        if file_mark in cfg.EVALUATE_MAP:
            midi = cfg.EVALUATE_MAP[file_mark]['midi']
            start_frame = cfg.EVALUATE_MAP[file_mark]['start_frame']
            fps = cfg.EVALUATE_MAP[file_mark]['fps']
            midi_offset = cfg.EVALUATE_MAP[file_mark]['midi_offset']

        model_files = list_models(args.model_dir)
        model_files.sort()  
        for model_file in model_files:
            if args.type=='white':
                visamthelper = VisAmtHelper(file_mark,midi,start_frame,fps,midi_offset,white_model=model_file,music_type=args.music_type)
            else:
                visamthelper = VisAmtHelper(file_mark,midi,start_frame,fps,midi_offset,black_model=model_file,music_type=args.music_type)
            visamthelper.process_img_dir(args.img_dir)
            if visamthelper.frame_result is not None:
                frame_result = visamthelper.frame_result 
                note_result = visamthelper.note_result 
                wf_rec,wf_prec,wf_F = frame_result['white']['recall'],frame_result['white']['precies'],frame_result['white']['F']
                bf_rec,bf_prec,bf_F = frame_result['black']['recall'],frame_result['black']['precies'],frame_result['black']['F']
                wn_rec,wn_prec,wn_F = note_result['white']['recall'],note_result['white']['precies'],note_result['white']['F']
                bn_rec,bn_prec,bn_F = note_result['black']['recall'],note_result['black']['precies'],note_result['black']['F']
                if args.type=='white':
                    print('{} frame white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(model_file,wf_prec,wf_rec,wf_F))
                else:
                    print('{} frame black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(model_file,bf_prec,bf_rec,bf_F))

    

