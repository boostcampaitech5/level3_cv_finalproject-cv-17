from setseed import set_seed

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coef(y_true, y_pred, smooth = 1e-6, RANDOM_SEED = 21):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    return (2. * intersection + smooth) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + smooth)

def IoU(y_true, y_pred, smooth = 1e-6, RANDOM_SEED = 21):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    union = torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return IoU