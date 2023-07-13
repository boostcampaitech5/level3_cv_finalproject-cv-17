import torch
import torch.nn.functional as F
from collections import OrderedDict

from tqdm.auto import tqdm

from metric import dice_coef
from setseed import set_seed

CLASSES = ['sclera', 'iris', 'pupil']

def validation(epoch, model, data_loader, criterion, RANDOM_SEED = 21, thr=0.5):
    set_seed(RANDOM_SEED)
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = 3
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            if type(outputs) == type(OrderedDict()):
                outputs = outputs['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            if type(criterion) == list:
                loss = 0
                for losses in criterion:
                    loss += losses[0](outputs, masks) * losses[1]
            else:
                loss = criterion(outputs, masks)

            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks, RANDOM_SEED)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)

    dice_dict = dict()
    for c, d in zip(CLASSES, dices_per_class):
        dice_dict[c] = d
    
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice