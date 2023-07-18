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
from torchvision import transforms

import sys

SEG_SAVE_PATH = "/opt/ml/eye_phone_streamlit/segmentation_results"

## RITnet ['eyelid', 'iris', 'pupil']

@st.cache_data
def load_seg_model(config_file) -> DenseNet2D:

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DenseNet2D(dropout=True,prob=0.2).to(device)
    # model = torch.load(config["model_path"])
    # model.to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    
    return model


def get_segmentation(model, image_paths, start_time):
    
    predicts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index, image_path in enumerate(image_paths):
        org_image = Image.open(image_path).convert("L")
        org_image = org_image.resize((192, 192))
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
        org_image = cv2.LUT(np.array(org_image), table)
        print(org_image.shape)
        
        # if self.transforms is not None:
        #     inputs = {"image": image}
        #     result = self.transforms(**inputs)
        #     image = result["image"]

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) # 안하면 결과 안좋음
        org_image = clahe.apply(np.array(np.uint8(org_image))) 

        # image = np.expand_dims(image, axis=0) #
        # image = torch.from_numpy(image).float()
        image = Image.fromarray(org_image) 
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((192,192)), # 위에서 먼저 하는게 더 잘 인식
            transforms.Normalize([0.5], [0.5])
            ])
        image = transform(image)
        image = image.unsqueeze(dim=0)
        print(image.shape)
        
        image = image.to(device) # image = image.cuda()
        output = model(image)
        print(output.shape)

        finish_time = time.time()
        print('segmentation time per image (image %d) : ' %(index+1), finish_time - start_time)
        start_time = finish_time

        if type(output) == type(OrderedDict()):
            output = output['out']

        bs,c,h,w = output.size()
        values, indices = output.cpu().max(1)
        result = indices.view(bs,h,w) # bs x h x w

        pred_img = result[0].cpu().numpy()/3.0
        # pred_img = result[0].cpu().numpy()
        # np.set_printoptions(threshold=sys.maxsize)
        print(pred_img.size)
        # print(pred_img)

        # inp = image[0].cpu().squeeze() * 0.5 + 0.5
        # img_org = np.clip(inp, 0,1)
        # img_org = np.array(img_org)
        # combine = np.hstack([img_org,pred_img])
        
        save_path = SEG_SAVE_PATH +'/'+ str(index) +'.jpg'

        # result_img = seg_visualize(org_image, pred_img)
        st.image(pred_img)
        # plt.imsave(save_path, combine)
        plt.imsave(save_path, pred_img)
        
        predicts.append(pred_img)
    
    return predicts


# def seg_visualize(image, result):
#     print(image.shape)
#     print(result.shape)
#     for i in range(192):
#         for j in range(192):
#             if int(result[i][j]*0.3) == 0: # eyelid
#                 image[0][i][j], image[1][i][j], image[2][i][j] = 255, 0, 0
#             if int(result[i][j]*0.3) == 1: # iris
#                 image[i][j][0], image[i][j][1], image[i][j][2] = 255, 0, 0
#             if int(result[i][j]*0.3) == 2: # pupil
#                 image[i][j][0], image[i][j][1], image[i][j][2] = 0, 255, 0

#     # image = Image.fromarray(image)
#     # image.show()
#     st.image(image)
#     return image