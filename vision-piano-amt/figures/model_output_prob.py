import matplotlib.pyplot as plt 
import os 
import numpy as np 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--file',type=str,default=None)
args = parser.parse_args()

def main():
    with open(args.file,'r') as fr:
        items = [l.strip() for l in fr.readlines()]
    probs = []
    for item in items:
        item = item.split('\t')
        prob = [float(data) for data in item]
        probs.append(prob)
    probs = np.array(probs)
    rows,cols = probs.shape 

    for col in range(cols):
        prob = probs[:,col]
        temp = np.where(prob>0.3)[0]
        if len(temp)==0:continue
        frames = np.arange(len(prob))
        plt.plot(frames,prob)
        plt.ylim(0.0,1)
        plt.show()


if __name__ =='__main__':
    main()
