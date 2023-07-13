import os
import cv2
import json
import torch
import numpy as np
import albumentations as A
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GroupKFold

from torch.utils.data import Dataset

from setseed import set_seed

class CustomDataset(Dataset):
    def __init__(self, mode = 'train', preprocess = None, augmentation = None, RANDOM_SEED = 21):
        set_seed(RANDOM_SEED)

        base_path = '/opt/ml/input/data'

        self.mode = mode
        
        if self.mode == 'train':
            self.data_path = base_path + '/train'
        elif self.mode == 'valid':
            self.data_path = base_path + '/valid'
        elif self.mode == 'test':
            self.data_path = base_path + '/test'
        else:
            raise Exception('mode 설정은 train/valid/test로만 가능합니다.')
        
        self.image_path = self.data_path + '/images'
        self.label_path = self.data_path + '/labels'

        self.image_names = os.listdir(self.image_path)
        self.label_names = os.listdir(self.label_path)

        assert len(self.image_names) == len(self.label_names), "이미지 개수와 라벨 개수가 다름"

        self.image_names.sort()  # 정렬하여 index가 같은 이미지와 라벨을 가르키도록 함
        self.label_names.sort()
        
        self.preprocess = preprocess
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, item):
        image_name = self.image_names[item]
        image_path = os.path.join(self.image_path, image_name)
        image = cv2.imread(image_path)
        image = np.array(image)
        image = np.pad(image, ((0,0),(0,16),(0,0)), 'constant', constant_values=0)

        label_name = self.label_names[item]
        label_path = os.path.join(self.label_path, label_name)
        label = np.load(label_path)
        label = np.pad(label, ((0,0),(0,16)), 'constant', constant_values=0)
        label = (np.arange(3) == label[...,None]-1).astype(np.uint8)

        if self.preprocess is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.preprocess(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        if self.mode == 'train' and self.augmentation is not None:
            image = np.array(image).astype(np.uint8)
            transform = A.Compose(self.augmentation)
            transformed = transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            image = np.array(image).astype(np.uint64)
            
        image = image.transpose(2, 0, 1)    # make channel first
        image = image / 255.
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        return image, label

def make_dataset(mode, preprocess = None, augmentation = None, RANDOM_SEED = 21):
    return CustomDataset(mode=mode, preprocess = preprocess, augmentation = augmentation, RANDOM_SEED = RANDOM_SEED)