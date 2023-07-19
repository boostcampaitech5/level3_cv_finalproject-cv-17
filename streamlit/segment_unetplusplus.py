from typing import Tuple

import streamlit as st
import torch
import yaml

from util_funcs import transform_image

import numpy as np
import cv2
from collections import OrderedDict
from PIL import Image
import time

import segmentation_models_pytorch as smp
from model import DenseNet2D
import matplotlib.pyplot as plt


SEG_SAVE_PATH = "/opt/ml/eye_phone_streamlit/segmentation_results"
MASK_SAVE_PATH = "/opt/ml/eye_phone_streamlit/mask_result"
## Unet++/mobilenet ['sclera', 'iris', 'pupil']

@st.cache_data
def load_seg_model(config_file):
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torch.load(config["model_path"])
    model.to(device) # model.cuda()
    # model.load_state_dict(torch.load(config["model_path"], map_location=device))
    
    return model


def get_segmentation(model, image_paths, start_time):

    predicts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index, image_path in enumerate(image_paths):
        # print('**3 channel**')
        org_image = cv2.imread(image_path)
        org_image = np.array(org_image) #
        print(org_image.shape)
        org_image = cv2.resize(org_image, (416,640), interpolation=cv2.INTER_CUBIC) #
        # org_image = np.pad(org_image, ((0,0),(0,16),(0,0)), 'constant', constant_values=0) #
        print(org_image.shape)
        # st.image(org_image)
        image = org_image / 255.
        print(org_image.shape)
        # st.image(org_image)

        # print('**1 channel**')
        # sec_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # sec_image = np.array(sec_image) #
        # print(sec_image.shape)
        # sec_image = cv2.resize(sec_image, (416,640), interpolation=cv2.INTER_CUBIC) #
        # # sec_image = np.pad(sec_image, ((0,0),(0,16),(0,0)), 'constant', constant_values=0) #
        # print(sec_image.shape)
        # st.image(sec_image)
        # image = sec_image / 255.
        # print(sec_image.shape)
        # st.image(sec_image)


        # if self.transforms is not None:
        #     inputs = {"image": image}
        #     result = self.transforms(**inputs)
        #     image = result["image"]

        image = image.transpose(2, 0, 1)
        # print(image.shape)
        image = np.expand_dims(image, axis=0) #
        # print(image.shape)
        image = torch.from_numpy(image).float()
        print(image.shape)
            
        image = image.to(device) # image = image.cuda()
        output = model(image)
        print(output.shape)

        finish_time = time.time()
        print('segmentation time per image (image %d) : ' %(index+1), finish_time - start_time)
        start_time = finish_time

        if type(output) == type(OrderedDict()):
            output = output['out']

        # outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
        n = torch.sigmoid(output) # output 변환
        n = (n > 0.5).detach().cpu().numpy()
        # print('#', n.size)
        # print('#', n)
        n = np.squeeze(n, axis = 0) #
        n = n.transpose(1, 2, 0) #
        n = n.astype(np.uint8) #
        

        save_path = SEG_SAVE_PATH +'/'+ str(index) +'.jpg'
        mask_save_path = MASK_SAVE_PATH + '/'+ str(index) +'.png'
        
        result_img = seg_visualize(org_image, n)
        plt.imsave(save_path, result_img)
        plt.imsave(mask_save_path, n)
        
        predicts.append(n)
    
    # return n
    return predicts


def seg_visualize(image, result):
    # print(image.shape)
    # print(result.shape)

    for i in range(640):
        for j in range(416):
            if result[i][j][0] == 1: # sclera, 공막, 적색
                result[i][j][0], result[i][j][1], result[i][j][2] = 255, 0, 0
                image[i][j][0], image[i][j][1], image[i][j][2] = 255, 0, 0
            if result[i][j][1] == 1: # iris, 홍채, 녹색
                result[i][j][0], result[i][j][1], result[i][j][2] = 0, 255, 0
                image[i][j][0], image[i][j][1], image[i][j][2] = 0, 255, 0
            if result[i][j][2] == 1: # pupil, 동공, 청색
                result[i][j][0], result[i][j][1], result[i][j][2] = 0, 0, 255
                image[i][j][0], image[i][j][1], image[i][j][2] = 0, 0, 255


    # image = Image.fromarray(image)
    # image.show()
    st.image(image)
    return image

