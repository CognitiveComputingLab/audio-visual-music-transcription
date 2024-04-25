import os 
import sys
import argparse 
from visAmthelper import VisAmtHelper 
from config import cfg

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',type=str,default=None)
parser.add_argument('--video',type=str,default=None)
parser.add_argument('--img_root',type=str,default=None)
parser.add_argument('--type',type=str,default='record',choices=['paper','record'])
args = parser.parse_args()


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

        visamthelper = VisAmtHelper(file_mark,midi,start_frame,fps,midi_offset,music_type=args.type)
        visamthelper.process_img_dir(args.img_dir)

    if args.video is not None:
        file_mark = os.path.basename(args.video).split('.')[0]
        midi,start_frame,fps,midi_offset = None,0,25,0
        if file_mark in cfg.EVALUATE_MAP:
            midi = cfg.EVALUATE_MAP[file_mark]['midi']
            start_frame = cfg.EVALUATE_MAP[file_mark]['start_frame']
            fps = cfg.EVALUATE_MAP[file_mark]['fps']
            midi_offset = cfg.EVALUATE_MAP[file_mark]['midi_offset']
        visamthelper = VisAmtHelper(file_mark,midi,start_frame,fps,midi_offset)
        visamthelper.process_video(args.video)
    
    if args.img_root is not None:
        img_dir_lists = [os.path.join(args.img_root,x) for x in os.listdir(args.img_root)]
        W_frame_recall,W_frame_precies,W_frame_F = 0,0,0
        B_frame_recall,B_frame_priecies,B_frame_F = 0,0,0
        W_note_recall,W_note_precies,W_note_F = 0,0,0
        B_note_recall,B_note_precies,B_note_F = 0,0,0
        img_dir_lists.sort()
        for img_dir in img_dir_lists:
            print(img_dir)
            save_dir = os.path.join(cfg.SAVE_IMG_DIR,os.path.basename(img_dir))
            #if os.path.exists(save_dir):continue
            if img_dir[-1]=='/':
                img_dir = img_dir[0:-1]
            file_mark = img_dir.split('/')[-1]
            midi,start_frame,fps,midi_offset = None,0,25,0
            if file_mark in cfg.EVALUATE_MAP:
                midi = cfg.EVALUATE_MAP[file_mark]['midi']
                start_frame = cfg.EVALUATE_MAP[file_mark]['start_frame']
                fps = cfg.EVALUATE_MAP[file_mark]['fps']
                midi_offset = cfg.EVALUATE_MAP[file_mark]['midi_offset']
            visamthelper = VisAmtHelper(file_mark,midi,start_frame,fps,midi_offset,music_type=args.type)     
            visamthelper.process_img_dir(img_dir)
            if visamthelper.frame_result is not None:
                frame_result = visamthelper.frame_result 
                note_result = visamthelper.note_result 
                wf_rec,wf_prec,wf_F = frame_result['white']['recall'],frame_result['white']['precies'],frame_result['white']['F']
                bf_rec,bf_prec,bf_F = frame_result['black']['recall'],frame_result['black']['precies'],frame_result['black']['F']
                wn_rec,wn_prec,wn_F = note_result['white']['recall'],note_result['white']['precies'],note_result['white']['F']
                bn_rec,bn_prec,bn_F = note_result['black']['recall'],note_result['black']['precies'],note_result['black']['F']
                W_frame_recall+=wf_rec 
                W_frame_precies+=wf_prec 
                W_frame_F+=wf_F 
                B_frame_recall+=bf_rec 
                B_frame_priecies+=bf_prec 
                B_frame_F+=bf_F 
                W_note_recall+=wn_rec 
                W_note_precies+=wn_prec 
                W_note_F+=wn_F 
                B_note_recall+=bn_rec 
                B_note_precies+=bn_prec 
                B_note_F+=bn_F 
        img_len = len(img_dir_lists)
        W_frame_recall,W_frame_precies,W_frame_F = W_frame_recall/img_len,W_frame_precies/img_len,W_frame_F/img_len 
        B_frame_recall,B_frame_priecies,B_frame_F = B_frame_recall/img_len,B_frame_priecies/img_len,B_frame_F/img_len 
        W_note_recall,W_note_precies,W_note_F = W_note_recall/img_len,W_note_precies/img_len,W_note_F/img_len 
        B_note_recall,B_note_precies,B_note_F = B_note_recall/img_len,B_note_precies/img_len,B_note_F/img_len 

        print('avg frame black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(B_frame_priecies,B_frame_recall,B_frame_F))
        print('avg frame white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(W_frame_precies,W_frame_recall,W_frame_F))
        print('avg note black\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(B_note_precies,B_note_recall,B_note_F))
        print('avg note white\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(W_note_precies,W_note_recall,W_note_F))

        # "D:\Dissertation Code\OMAPS2\test\video\01_01.mp4"