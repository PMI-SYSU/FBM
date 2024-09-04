#Fromhttps://blog.csdn.net/weixin_40522523/article/details/82823812
#The file to load MNIST dataset
#from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np
from struct import unpack
import os
import random

#TRAIN_IMAGES = str(os.getcwd())+'/mnist/train-images.idx3-ubyte'
#TRAIN_LABELS = str(os.getcwd())+'/mnist/train-labels.idx1-ubyte'
#TEST_IMAGES = str(os.getcwd())+'/mnist/t10k-images.idx3-ubyte'
#TEST_LABELS = str(os.getcwd())+'/mnist/t10k-labels.idx1-ubyte'
TRAIN_IMAGES = 'D:\Japan\work\G_map\Code\data\MNIST\train-images.idx3-ubyte'
TRAIN_LABELS = 'D:\Japan\work\G_map\Code\data\MNIST\train-labels.idx1-ubyte'
TEST_IMAGES = 'D:\Japan\work\G_map\Code\data\MNIST\t10k-images.idx3-ubyte'
TEST_LABELS = 'D:\Japan\work\G_map\Code\data\MNIST\t10k-labels.idx1-ubyte'

def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[int(label[i])] = 1
    return lab

def __one_hot_label_sort(label, num):
    lab = np.zeros((label.size, num))
    for i, row in enumerate(lab):
        row[int(label[i])] = 1
    return lab


def load_mnist(train_image_path=TRAIN_IMAGES, train_label_path=TRAIN_LABELS, test_image_path=TEST_IMAGES, test_label_path=TEST_LABELS, normalize=True, one_hot=False):
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            #print('label[key]:', label[key].shape)
            label[key] = __one_hot_label(label[key])#(batch,)->(batch,10)
            #print('label[key]:', label[key].shape)
            #print('image[key]:', image[key].shape)
            
    return (image['train'], label['train']), (image['test'], label['test'])#[num, dim]


def load_mnist_sort(sort_label, num_train, num_test, train_image_path=TRAIN_IMAGES, train_label_path=TRAIN_LABELS, test_image_path=TEST_IMAGES, test_label_path=TEST_LABELS, normalize=True, one_hot=False):
    #返回值内image_sort_num[train]={ 'label':np.array(num_train, dim) }， label_sort_num['train']={ 'label':np.array(num_train, dim) }
    
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])
           
 
    sort_num = {
        'train' : num_train,
        'test'  : num_test
    }
    
    image_sort_num = {
        'train' : 0,
        'test'  : 0   
        }#存放不同标签的数据字典
    
    label_sort_num = {
        'train' : 0,
        'test'  : 0  
        }
    
    
    for key in ('train', 'test'):
        #完成对不同标签数据组成字典：image_sort_num[train]={ 'label':np.array(num_train, dim) }
        Data_label = {str(x): [] for x in sort_label} #[[],[],[],[],[],[],[],[],[],[]]
        num_label ={str(x): 0 for x in sort_label} 
        index_label = {str(x): 1 for x in sort_label} 
        
        all_data = image[key]#[num_total, dim]
        all_label = label[key]#[num_total]        
        
        for i in range(len(all_data)):#(len(all_data)):
            #print(np.prod(sort_label-all_label[i]))
            if np.prod(sort_label-all_label[i])!= 0:
                continue
            else:
                #print('num_label[str(all_label[i])]',num_label[str(all_label[i])])
                if num_label[str(all_label[i])]==sort_num[key]:
                    index_label[str(all_label[i])] = index_label[str(all_label[i])]*0
                else:
                    Data_label[str(all_label[i])].append(all_data[i])
                    num_label[str(all_label[i])]+=1
                if sum(index_label.values())==0:
                    break
        for k in Data_label.keys():
            Data_label[k] = np.array(Data_label[k])
            
        image_sort_num[key]=Data_label
    
    for key in ('train', 'test'):
        label = {}
        for i in sort_label:
            label[str(i)] = np.ones(sort_num[key])*sort_label.index(i)#不在乎真实标签数值，而是其标签排序
            if one_hot:
                label[str(i)] = __one_hot_label_sort(label[str(i)], len(sort_label))
                
        label_sort_num[key] = label
    
    return (image_sort_num['train'], label_sort_num['train']), (image_sort_num['test'], label_sort_num['test'])


    
    