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

def train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS = 30, VAL_EVERY = 1, folder_name = 'last_model', RANDOM_SEED = 21,PATIENCE=5):
    print(f'Start training..')
    set_seed(RANDOM_SEED)

    """
    if not os.path.isdir(os.path.join(SAVED_DIR, folder_name)):                                                           
        os.mkdir(os.path.join(SAVED_DIR, folder_name))
    else:
        idx = 2
        print()
        while os.path.isdir(os.path.join(SAVED_DIR, folder_name) + '_' + str(idx)):
            idx += 1 
        folder_name = folder_name + '_' + str(idx)
        os.mkdir(os.path.join(SAVED_DIR, folder_name))
    print("result model will be saved in {}".format(os.path.join(SAVED_DIR, folder_name)))
    """

    print("result model will be saved in {}".format(SAVED_DIR))
    
    n_class = 3
    best_dice = 0.
    best_epoch = 0
    loss = 0
    early_stop=0
    
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
            if (step + 1) % 100 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion, RANDOM_SEED, 0.5)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {os.path.join(SAVED_DIR, folder_name)}")
                best_dice = dice
                best_epoch = epoch + 1
                early_stop=0
                save_best_model(model, folder_name)
            else:
                print('No update')
                print(f"Best performance was at epoch: {best_epoch}, {best_dice:.4f}")
                early_stop+=1
                if early_stop>=PATIENCE and best_dice>0.5:
                    print("No more update")
                    break
    return folder_name

def save_best_model(model, folder_name):
    output_path = SAVED_DIR
    if not os.path.isdir(output_path):                                                           
        os.mkdir(output_path)
    torch.save(model, output_path + 'best.pt')