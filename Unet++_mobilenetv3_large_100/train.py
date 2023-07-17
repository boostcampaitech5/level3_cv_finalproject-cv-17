from validation import validation
from setseed import set_seed

import os
import torch
import numpy as np
import random
from collections import OrderedDict
import datetime

SAVED_DIR = "/opt/ml/input/code/trained_model/"
if not os.path.isdir(SAVED_DIR):                                                           
    os.mkdir(SAVED_DIR)

def train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS = 30, VAL_EVERY = 1, folder_name = 'last_model', RANDOM_SEED = 21):
    print(f'Start training..')
    set_seed(RANDOM_SEED)

    CLASSES = ['sclera', 'iris', 'pupil']
    
    if not os.path.isdir(os.path.join(SAVED_DIR, folder_name)):
        output_path = os.path.join(SAVED_DIR, folder_name)
        os.mkdir(output_path)
    else:
        idx = 2
        while os.path.isdir(os.path.join(SAVED_DIR, folder_name) + '_' + str(idx)):
            idx += 1 
        folder_name = folder_name + '_' + str(idx)
        output_path = os.path.join(SAVED_DIR, folder_name)
        os.mkdir(output_path)

    print("result model will be saved in {}".format(output_path))
    
    n_class = 3
    
    best_pupil_iou = 0
    best_pupil_dice = 0
    best_iou = 0
    best_dice = 0

    best_pupil_iou_epoch = 0
    best_pupil_dice_epoch = 0
    best_iou_epoch = 0
    best_dice_epoch = 0
    
    loss = 0
    
    for epoch in range(NUM_EPOCHS):
        print(
            f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
            f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
            f'Step [0/{len(train_loader)}]'
        )
        
        model.train()

        for step, (images, masks) in enumerate(train_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)
            if type(outputs) == type(OrderedDict()):
                outputs = outputs['out']
            
            # loss 계산
            if type(criterion) == list:
                loss = 0
                for losses in criterion:
                    loss += losses[0](outputs, masks) * losses[1]
            else:
                loss = criterion(outputs, masks)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 150 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),5)}'
                )
                
        if (epoch + 1) % VAL_EVERY == 0:
            pupil_iou, pupil_dice, iou, dice = validation(epoch + 1, model, val_loader, criterion, RANDOM_SEED, 0.5)
            print()
            
            if best_pupil_iou < pupil_iou:
                best_pupil_iou = pupil_iou
                best_pupil_iou_epoch = epoch + 1
                print(f"new best pupil iou score: {best_pupil_iou:.5f}")
                torch.save(model, output_path + '/best_pupil_iou.pt')
            else:
                print(f"best pupil iou score was at {best_pupil_iou_epoch}")
                
            if best_pupil_dice < pupil_dice:
                best_pupil_dice = pupil_dice
                best_pupil_dice_epoch = epoch + 1
                print(f"new best pupil dice score: {best_pupil_dice:.5f}")
                torch.save(model, output_path + '/best_pupil_dice.pt')
            else:
                print(f"best pupil dice score was at {best_pupil_dice_epoch}")

            if best_iou < iou:
                best_iou = iou
                best_iou_epoch = epoch + 1
                print(f"new best mean iou score: {best_iou:.5f}")
                torch.save(model, output_path + '/best_mean_iou.pt')
            else:
                print(f"best iou score was at {best_iou_epoch}")

            if best_dice < dice:
                best_dice = dice
                best_dice_epoch = epoch + 1
                print(f"new best mean dice score: {best_dice:.5f}")
                torch.save(model, output_path + '/best_mean_dice.pt')
            else:
                print(f"best dice score was at {best_dice_epoch}")
            print()

    print(f"best pupil iou score: {best_pupil_iou:.5f} was at epoch {best_pupil_iou_epoch}\n")
    print(f"best pupil dice score: {best_pupil_dice:.5f} was at epoch {best_pupil_dice_epoch}\n")
    print(f"best mean iou score: {best_iou:.5f} was at epoch {best_iou_epoch}\n")
    print(f"best mean dice score: {best_dice:.5f} was at epoch {best_dice_epoch}\n")
    print()

    log = open(output_path + "/log.txt", "w")
    log.write(f"epoch: {NUM_EPOCHS}\n")
    log.write(f"loss: {criterion}\n")
    log.write(f"optimizer: {optimizer}\n")
    log.write(f"best pupil iou score: {best_pupil_iou:.5f} was at epoch {best_pupil_iou_epoch}\n")
    log.write(f"best pupil dice score: {best_pupil_dice:.5f} was at epoch {best_pupil_dice_epoch}\n")
    log.write(f"best mean iou score: {best_iou:.5f} was at epoch {best_iou_epoch}\n")
    log.write(f"best mean dice score: {best_dice:.5f} was at epoch {best_dice_epoch}\n")
    log.close()

    return output_path