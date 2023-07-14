#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:47:44 2019

@author: Aayush

This file contains the dataloader and the augmentations and preprocessing done

Required Preprocessing for all images (test, train and validation set):
1) Gamma correction by a factor of 0.8
2) local Contrast limited adaptive histogram equalization algorithm with clipLimit=1.5, tileGridSize=(8,8)
3) Normalization
    
Train Image Augmentation Procedure Followed 
1) Random horizontal flip with 50% probability.
2) Starburst pattern augmentation with 20% probability. 
3) Random length lines augmentation around a random center with 20% probability. 
4) Gaussian blur with kernel size (7,7) and random sigma with 20% probability. 
5) Translation of image and labels in any direction with random factor less than 20.
"""

import numpy as np
import torch
from torch.utils.data import Dataset 
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
import os.path as osp
from utils import one_hot2dist
import copy

from sklearn.model_selection import GroupKFold
import xml.etree.ElementTree as ET

transform = transforms.Compose(
    
    [transforms.ToTensor(),
    #  transforms.Resize((192,192)),
     transforms.Normalize([0.5], [0.5])])
  
#%%
class RandomHorizontalFlip(object):
    def __call__(self, img,label):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT),\
                        label.transpose(Image.FLIP_LEFT_RIGHT)
        return img,label
    
class Starburst_augment(object):
    ## We have generated the starburst pattern from a train image 000000240768.png
    ## Please follow the file Starburst_generation_from_train_image_000000240768.pdf attached in the folder 
    ## This procedure is used in order to handle people with multiple reflections for glasses
    ## a random translation of mask of starburst pattern
    def __call__(self, img):
        x=np.random.randint(1, 40)
        y=np.random.randint(1, 40)
        mode = np.random.randint(0, 2)
        starburst=Image.open('starburst_black.png').convert("L")
        if mode == 0:
            starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
            starburst = starburst[:, :-x]
        if mode == 1:
            starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
            starburst = starburst[:, x:]

        img[92+y:549+y,0:400]=np.array(img)[92+y:549+y,0:400]*((255-np.array(starburst))/255)+np.array(starburst)
        return Image.fromarray(img)

def getRandomLine(xc, yc, theta):
    x1 = xc - 50*np.random.rand(1)*(1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*np.random.rand(1) + 50)*(1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    return x1, y1, x2, y2

class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value=np.random.randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img,(7,7),sigma_value))

class Translation(object):
    def __call__(self, base,mask):
        factor_h = 2*np.random.randint(1, 20)
        factor_v = 2*np.random.randint(1, 20)
        mode = np.random.randint(0, 4)
#        print (mode,factor_h,factor_v)
        if mode == 0:
            aug_base = np.pad(base, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_base = aug_base[:-factor_v, :]
            aug_mask = aug_mask[:-factor_v, :]
        if mode == 1:
            aug_base = np.pad(base, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_base = aug_base[factor_v:, :]
            aug_mask = aug_mask[factor_v:, :]
        if mode == 2:
            aug_base = np.pad(base, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_base = aug_base[:, :-factor_h]
            aug_mask = aug_mask[:, :-factor_h]
        if mode == 3:
            aug_base = np.pad(base, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_base = aug_base[:, factor_h:]
            aug_mask = aug_mask[:, factor_h:]
        return Image.fromarray(aug_base), Image.fromarray(aug_mask)     
            
class Line_augment(object):
    def __call__(self, base):
        yc, xc = (0.3 + 0.4*np.random.rand(1))*base.shape
        aug_base = copy.deepcopy(base)
        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       
        
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       

  
class EyeDataset(Dataset):
    def __init__(self, filepath, split='train',transform=None,**args):
        self.transform = transform
        self.filepath= osp.join(filepath,split)
        self.split = split
        listall = []
        
        for file in os.listdir(osp.join(self.filepath,'images')):   
            if file.endswith(".png"):
               listall.append(file.strip(".png"))
        self.list_files=listall

        self.testrun = args.get('testrun')
        
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    def __len__(self):
        if self.testrun:
            return 10
        return len(self.list_files)

    def __getitem__(self, idx):
        imagepath = osp.join(self.filepath,'images',self.list_files[idx]+'.png')
        pilimg = Image.open(imagepath).convert("L")
        H, W = pilimg.width , pilimg.height
       
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #Fixed gamma value for      
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
        pilimg = cv2.LUT(np.array(pilimg), table)
        

        if self.split != 'test':
            labelpath = osp.join(self.filepath,'labels',self.list_files[idx]+'.png.npy')
            label = np.load(labelpath)    
            label = np.resize(label,(W,H))
            label = label.astype((np.uint8))
            label = Image.fromarray(label)     
               
        if self.transform is not None:
            if self.split == 'train':   
                if random.random() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))   
                if random.random() < 0.4:
                    pilimg, label = Translation()(np.array(pilimg),np.array(label))
                
        img = self.clahe.apply(np.array(np.uint8(pilimg)))    
        img = Image.fromarray(img)      
            
        if self.transform is not None:
            if self.split == 'train':
                img, label = RandomHorizontalFlip()(img,label)
            img = self.transform(img)    


        if self.split != 'test':
            ## This is for boundary aware cross entropy calculation
            spatialWeights = cv2.Canny(np.array(label),0,3)/255
            spatialWeights=cv2.dilate(spatialWeights,(3,3),iterations = 1)*20
            
            ##This is the implementation for the surface loss
            # Distance map for each class
            distMap = []
            for i in range(0, 4):
                distMap.append(one_hot2dist(np.array(label)==i))
            distMap = np.stack(distMap, 0)           
#            spatialWeights=np.float32(distMap) 
            
            
        if self.split == 'test':
            ##since label, spatialWeights and distMap is not needed for test images
            return img,0,self.list_files[idx],0,0
            
        label = MaskToTensor()(label)
        return img, label, self.list_files[idx],spatialWeights,np.float32(distMap) 
    
class NeWEyeDataset(Dataset):
    def __init__(self,split = 'train',transform=None,**args):
        self.transform = transform
        self.filepath = osp.join
        self.split = split
        self.all_img = []
        
        
        _filenames = np.array(pngs)
        _labelnames = np.array(xmls)
        
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        ys = [0 for fname in _filenames]
        
        # split train-valid
        print(len(list(set(groups))))
        # dummy label
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.split == 'train':
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                filenames += list(_filenames[y])
            else:
                filenames = list(_filenames[y])
                
                # skip i > 0
                break
            
        self.filenames = filenames
        self.labelnames = _labelnames
    def __len__(self):
            return len(self.filenames)

    def __getitem__(self, idx):
        # 이미지 받기
        image_path = self.filenames[idx]
        image =  cv2.imread(image_path)
       
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #Fixed gamma value for      
        # table = 255.0*(np.linspace(0, 1, 256)**0.8)
        # image = cv2.LUT(np.array(image), table)
        
        # 라벨받기
        
        labelpath = self.labelnames[idx]
        tree = ET.parse(labelpath)
        
        label_shape = tuple(image.shape[:2]) + (3, )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        attrs = tree.findall(f'./image[@name="{os.path.basename(image_path)}"]/polygon[@label]')
        
        for attr in attrs:
            points = attr.get('points').split(';')
            points = [list(map(float, point.split(','))) for point in points]
            # print(points)
            
            class_name = attr.get('label')
            class_ind = CLASS2IND[class_name]
            points = np.array(points).astype(np.int32)
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        image = image.transpose(2, 0, 1)    # make channel first
        image = image / 255.
        # label = label.transpose(2, 0, 1)
        label = Image.fromarray(label)
        
        if self.split != 'test':
            ## This is for boundary aware cross entropy calculation
            spatialWeights = cv2.Canny(np.array(label),0,3)/255
            spatialWeights=cv2.dilate(spatialWeights,(3,3),iterations = 1)*20
            
            ##This is the implementation for the surface loss
            # Distance map for each class
            distMap = []
            for i in range(0, 4):
                distMap.append(one_hot2dist(np.array(label)==i))
            distMap = np.stack(distMap, 0)           
            spatialWeights=np.float32(distMap) 

        image = Image.fromarray(image)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()  
            
        return image, label ,spatialWeights,np.float32(distMap) 


IMAGE_ROOT = "/opt/ml/data_input/images"   
XML_ROOT =  "/opt/ml/data_input/label"

CLASSES = ["right_pupil","right_eyelid","right_iris"]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

pngs = []
for folder in os.listdir(IMAGE_ROOT):
    for file in os.listdir(IMAGE_ROOT +"/"+ folder+"/VR/IR"):
        if file[-3:].lower() == "png":
            file_path = IMAGE_ROOT + "/"+folder +"/VR/IR/"+ file
            pngs.append(file_path)
pngs.sort()

xmls = []
for folder in os.listdir(XML_ROOT):
    for file in os.listdir(XML_ROOT +"/"+ folder+"/VR"):
        if file[-3:].lower() == "xml":
            file_path = XML_ROOT + "/"+folder +"/VR/"+ file
            xmls.append(file_path)
xmls.sort()     
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = EyeDataset(split='train',transform=transform)
    img, label= ds[0]
    print(img.shape,label.shape)
    # plt.subplot(121)
    # plt.imshow(np.array(label))
    # plt.subplot(122)
    # plt.imshow(np.array(img)[0,:,:],cmap='gray')
