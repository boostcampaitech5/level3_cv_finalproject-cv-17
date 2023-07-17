from typing import Tuple

import streamlit as st
import torch
import yaml

from utils import transform_image

import segmentation_models_pytorch as smp
import numpy as np
import cv2
from collections import OrderedDict
from PIL import Image
import time


SAVE_PATH = "/opt/ml/eye_phone_streamlit/detection_results"

@st.cache_data
def load_seg_model(config_file):
    
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = torch.load(config["model_path"])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(config["model_path"], map_location=device))
    
    return model


## Unet++/mobilenet ['sclera, 'iris', 'pupil'] / 
def get_segmentation(model, image, start_time):
    image = np.array(image) #
    image = cv2.resize(image, (416,640), interpolation=cv2.INTER_CUBIC) #
    # image = np.pad(image, ((0,0),(0,16),(0,0)), 'constant', constant_values=0) #
    org_image = image # for visualization
    image = image / 255.

    # if self.transforms is not None:
    #     inputs = {"image": image}
    #     result = self.transforms(**inputs)
    #     image = result["image"]

    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0) #
    image = torch.from_numpy(image).float()
        
    
    image = image.cuda()
    output = model(image)

    finish_time = time.time()
    print('segmentation time per image (segment each image from detection) : ', finish_time - start_time)

    if type(output) == type(OrderedDict()):
        output = output['out']

    # outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
    n = torch.sigmoid(output) # output 변환
    n = (n > 0.5).detach().cpu().numpy()

    n = np.squeeze(n, axis = 0) #
    n = n.transpose(1, 2, 0) #
    n = n.astype(np.uint8) #
    # print(n)
    seg_visualize(org_image, n)
    
    return n, finish_time


def seg_visualize(image, result):

    print(image.shape)

    for i in range(640):
        for j in range(416):
            if result[i][j][0] == 1: # sclera, 공막, 적색
                image[i][j][0], image[i][j][1], image[i][j][2] = 255, 0, 0
            if result[i][j][1] == 1: # iris, 홍채, 녹색
                image[i][j][0], image[i][j][1], image[i][j][2] = 0, 255, 0
            if result[i][j][2] == 1: # pupil, 동공, 청색
                image[i][j][0], image[i][j][1], image[i][j][2] = 0, 0, 255

    # image = Image.fromarray(image)
    # image.show()
    st.image(image)

