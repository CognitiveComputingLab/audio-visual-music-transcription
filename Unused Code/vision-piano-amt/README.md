## 基于视觉的钢琴转录算法实现

### Description
该工程是基于视觉的钢琴转录算法，相比与之前的方法更加稳定，精度更高，可以处理各个视角下的钢琴转录，主要包括几个模块，键盘分割模块，按键定位模块，手的分割定位模块，钢琴按键按下分类模块等，包含的技术有语义分割，图像分类和传统的视觉算法，在公开数据集上white fscore=0.96,black fscore=0.98。

### Requirement
1. python-opencv
2. pytorch>=1.1.0
3. easydict

### module 
*	键盘分割采用的pspnet,在3rdparty/segmentaion下
*	钢琴按键分类在3rdparty/key_classification
*	其他的转录代码有main.py执行

### run
```
python main.py --img_dir video_file/img_path
```


## Implementation of vision-based piano transcription algorithm

### Description
This project is a vision-based piano transcription algorithm. It is more stable and more accurate than previous methods. It can handle piano transcription from various perspectives. It mainly includes several modules, keyboard segmentation module, key positioning module, and hand segmentation and positioning. module, piano key press classification module, etc., including technologies such as semantic segmentation, image classification and traditional visual algorithms. On the public data set, white fscore=0.96, black fscore=0.98.

### Requirements
1.python-opencv
2.pytorch>=1.1.0
3. easydict

### module
* Keyboard segmentation uses pspnet, under 3rdparty/segmentaion # Create weights and data for these two
* Piano key classification is in 3rdparty/key_classification
* Other transcription codes are executed in main.py

### run
```
python main.py --img_dir video_file/img_path
```