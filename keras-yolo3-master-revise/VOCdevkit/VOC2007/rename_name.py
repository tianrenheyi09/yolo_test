import os
import numpy as np
from os import getcwd
import re

#############修改图片名称
wd = getcwd()
path = 'E:\object_detection\keras-yolo3-master-revise\VOCdevkit\VOC2007\JPEGImages'
fold_list = os.listdir(path)

for fileNmae in fold_list:
    temp = re.findall(r"\d+",fileNmae)
    print(temp)
    old_path = os.path.join(os.path.abspath(path), fileNmae)
    new_path = os.path.join(os.path.abspath(path),'0000'+temp[0] + '.jpg')
    os.rename(old_path,new_path)

#######修改xml名称

path = 'E:\object_detection\keras-yolo3-master\VOCdevkit\VOC2012\Annotations'
fold_list = os.listdir(path)

for fileNmae in fold_list:
    temp = re.findall(r"\d+",fileNmae)
    print(temp)
    old_path = os.path.join(os.path.abspath(path), fileNmae)
    new_path = os.path.join(os.path.abspath(path),temp[0] + '.xml')
    os.rename(old_path,new_path)

###############修改test  train的值
import pandas as pd

x_path = "E:\object_detection\keras-yolo3-master\VOCdevkit\VOC2012\ImageSets\Main"
x = pd.read_csv(x_path+'\\test.txt',header=None,delimiter=' ',dtype=str)
del x[0]
x.to_csv(x_path+'\\test1.txt',index=False,header=None)

x = pd.read_csv(x_path+'\\train.txt',header=None,delimiter=' ',dtype=str)
del x[0]
x.to_csv(x_path+'\\train1.txt',index=False,header=None)

x = pd.read_csv(x_path+'\\trainval.txt',header=None,delimiter=' ',dtype=str)
del x[0]
x.to_csv(x_path+'\\trainval1.txt',index=False,header=None)

x = pd.read_csv(x_path+'\\val.txt',header=None,delimiter=' ',dtype=str)
del x[0]
x.to_csv(x_path+'\\val1.txt',index=False,header=None)