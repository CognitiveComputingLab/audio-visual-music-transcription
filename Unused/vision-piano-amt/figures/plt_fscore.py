import os 
import matplotlib.pyplot as plt 
import numpy as np 
from IPython import embed 

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False 
#plt.rcParams['figure.figsize'] = (1920, 1080)

cfg = {
        'save_dir':'/home/data/lj/Piano/saved/experment/network/conv3net/'
}
colors = [
        'r','g','b','darkorange','violet','coral','peru','lightpink','tan','orangered','sage','c','nvay','skyblue','darkred','m'
]

class PlotBase(object):
    def __init__(self,plt_path):
        self.init_config(plt_path)
        
        #self.plt_figure_by_light()
        self.plt_figure_by_file()

    def init_config(self,plt_path):
        self.plt_marks = list()
        self.get_plt_marks(plt_path)
        self.frame_white_precies = dict()
        self.frame_white_recall = dict()
        self.frame_white_fscore = dict()

        self.frame_black_precies = dict()
        self.frame_black_recall = dict()
        self.frame_black_fscore = dict()

        self.note_white_precies = dict()
        self.note_white_recall = dict()
        self.note_white_fscore = dict()

        self.note_black_precies = dict()
        self.note_black_recall = dict()
        self.note_black_fscore = dict()

        self.note_total_precies = dict()
        self.note_total_recall = dict()
        self.note_total_fscore = dict()

        self.get_evalresult()
    
    def get_number_level(self,number):
        if number>=100 and number<=200:return '100-200'
        elif number>200 and number<=300:return '200-300'
        elif number>300 and number<=400:return '300-400'
        elif number>400 and number<=500:return '400-500'
        elif number>500 and number<=600:return '500-600'
        elif number>600 and number<=700:return '600-700'
        elif number>700 and number<=800:return '700-800'
        elif number>=800 and number<900:return '800-900'
        else:raise ValueError('number is wrong')

    def split_data_helper(self,splitdata):
        splitdata = sorted(splitdata.items(), key=lambda d: d[0])
        light_left_data = {'100-200':0,'200-300':0,'300-400':0,'400-500':0,
                            '500-600':0,'600-700':0,'700-800':0,'800-900':0}
        light_middle_data =  {'100-200':0,'200-300':0,'300-400':0,'400-500':0,
                            '500-600':0,'600-700':0,'700-800':0,'800-900':0}
        light_right_data =  {'100-200':0,'200-300':0,'300-400':0,'400-500':0,
                            '500-600':0,'600-700':0,'700-800':0,'800-900':0}

        light_left_data = {}
        light_middle_data = {}
        light_right_data = {}
        for key,value in splitdata:
            if 'baseline' in key:
                index = self.plt_marks.index(key)
                if 'left' in self.plt_marks[index+1]:light_left_data['0-100'] = value 
                elif 'right' in self.plt_marks[index+1]:light_right_data['0-100'] = value 
                else:light_middle_data['0-100'] = value 
            else:
                number = int(key.split('_')[-1])
                nlevel = self.get_number_level(number)
                if 'left' in key:light_left_data[nlevel] = value
                elif 'right' in key: light_right_data[nlevel] = value
                else: light_middle_data[nlevel] = value 
        return light_left_data,light_middle_data,light_right_data

    def split_data_by_light(self): 
        self.plt_marks.sort()
        self.light_left_frame_white,self.light_middle_frame_white,self.light_right_frame_white = \
                    self.split_data_helper(self.frame_white_fscore)
        self.light_left_frame_black,self.light_middle_frame_black,self.light_right_frame_black = \
                    self.split_data_helper(self.frame_black_fscore)
        self.light_left_note_white,self.light_middle_note_white,self.light_right_note_white = \
                    self.split_data_helper(self.note_white_fscore)
        self.light_left_note_black,self.light_middle_note_black,self.light_right_note_black = \
                    self.split_data_helper(self.note_black_fscore)

    def plt_figure_by_light(self):
        self.split_data_by_light()
        figure_datas = {
            #'right_frame_white':self.light_right_frame_white, 
            #'left_frame_white':self.light_left_frame_white,
            #'middle_frame_white':self.light_middle_frame_white, 
            'right_frame_black':self.light_right_frame_black, 
            'left_frame_black':self.light_left_frame_black, 
            'middle_frame_black':self.light_middle_frame_black, 
        }
        count = 0 
        for label,figure_data in figure_datas.items():
            xlabels = figure_data.keys()
            ylabels = figure_data.values()
            l1,= plt.plot(xlabels,ylabels,'cx--',color=colors[count],marker='o',label=label)
            count+=1 
        plt.ylim(0.0, 1)
        plt.gcf().set_facecolor(np.ones(3))
        plt.grid()
        plt.xlabel('Video Sample')
        plt.ylabel('F1Score')
        y_ticks = np.arange(0, 1.1, 0.1)
        plt.yticks(y_ticks)
        plt.xticks(fontsize=7)
        plt.legend(loc='best')
        plt.title('Performance', color='g')
        plt.show()

    def plt_figure_by_file(self):
        figure_datas = {
                'frame_black_fscore':self.frame_black_fscore,
                #'frame_black_precies':self.frame_black_precies,
                #'frame_black_recall':self.frame_black_recall,
                
                'frame_white_fscore':self.frame_white_fscore,
                #'frame_white_precies':self.frame_white_precies,
                #'frame_white_recall':self.frame_white_recall,

                #'note_black_fscore':self.note_black_fscore,
                #'note_black_precies':self.note_black_precies,
                #'note_black_recall':self.note_black_recall,

                #'note_white_fscore':self.note_white_fscore,
                #'note_white_precies':self.note_white_precies,
                #'note_white_recall':self.note_white_recall,

                #'note_white_fscore':self.note_total_fscore,
                #'note_total_precies':self.note_total_precies,
                #'note_total_recall':self.note_total_recall
        }
        count = 0
        for label,figure_data in figure_datas.items():
            figure_data = sorted(figure_data.items(), key=lambda d: d[0])
            xlabels = [x[0] for x in figure_data]
            ylabels = [x[1] for x in figure_data]

            l1,= plt.plot(xlabels,ylabels,'cx--',color=colors[count],marker='o',label=label)
            count+=1 

        plt.ylim(0.0, 1)
        plt.gcf().set_facecolor(np.ones(3))
        plt.grid()
        plt.xlabel('VideoSample')
        plt.ylabel('FScore')
        y_ticks = np.arange(0, 1.1, 0.1)
        plt.yticks(y_ticks)
        plt.xticks(fontsize=7)
        plt.legend(loc='best')
        plt.title('Performance', color='g')
        plt.show()

    def get_plt_marks(self,plt_path):
        subpaths = [os.path.join(plt_path,x) for x in os.listdir(plt_path)]
        for subpath in subpaths:
            if os.path.isdir(subpath):
                self.get_plt_marks(subpath)
            else:
                if subpath.endswith('jpg'):
                    mark = subpath.split('/')[-2]
                    if mark not in self.plt_marks:
                        self.plt_marks.append(mark)
    
    def get_evalresult(self):
        img_dirs = [os.path.join(cfg['save_dir'],x) for x in self.plt_marks]
        img_dirs = [x for x in img_dirs if os.path.exists(x)]
        eval_files = [os.path.join(x,'evalresult.txt') for x in img_dirs]
        
        for i,eval_file in enumerate(eval_files):
            if not os.path.exists(eval_file):continue 
            mark = self.plt_marks[i]
            with open(eval_file,'r') as fr:
                items = fr.readlines()
            assert len(items)==5,'eval lines is wrong'
            for idx,item in enumerate(items):
                item = item.strip().split('\t')
                assert len(item)==4,'eval data is wrong'
                item = item[1:]
                for idy,data in enumerate(item):
                    val = float(data.split(':')[-1])
                    if idx==0:
                        if idy==0:self.frame_black_precies[mark] = val  
                        elif idy==1:self.frame_black_recall[mark] = val 
                        else:self.frame_black_fscore[mark] = val 
                    elif idx==1:
                        if idy==0:self.frame_white_precies[mark] = val  
                        elif idy==1:self.frame_white_recall[mark] = val 
                        else:self.frame_white_fscore[mark] = val
                    elif idx==2:
                        if idy==0:self.note_black_precies[mark] = val  
                        elif idy==1:self.note_black_recall[mark] = val 
                        else:self.note_black_fscore[mark] = val
                    elif idx==3:
                        if idy==0:self.note_white_precies[mark] = val  
                        elif idy==1:self.note_white_recall[mark] = val 
                        else:self.note_white_fscore[mark] = val
                    else:
                        if idy==0:self.note_total_precies[mark] = val  
                        elif idy==1:self.note_total_recall[mark] = val 
                        else:self.note_total_fscore[mark] = val 



if __name__=='__main__':
    plt_path = '/home/data/lj/Piano/test_imgs/Record/1225'
    #plt_path = '/home/data/lj/Piano/paperData/IWSSIP/TestSet/images'
    pltbase = PlotBase(plt_path)
    
