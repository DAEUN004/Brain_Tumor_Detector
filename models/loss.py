'''
@article{focal-unet,
  title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}
'''

import torch
import torch.nn.functional as F

epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    bce_loss = F.binary_cross_entropy(y_pred, y_true)
    dice_loss_value = dice_loss(y_true, y_pred)
    return bce_loss + dice_loss_value

def confusion(y_true, y_pred):
    y_pred_pos = torch.clamp(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.clamp(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = torch.sum(y_pos * y_pred_pos)
    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall

def tp(y_true, y_pred):
    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pos = torch.round(torch.clamp(y_true, 0, 1))
    true_pos = (torch.sum(y_pos * y_pred_pos) + smooth) / (torch.sum(y_pos) + smooth)
    return true_pos

def tn(y_true, y_pred):
    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.round(torch.clamp(y_true, 0, 1))
    y_neg = 1 - y_pos
    true_neg = (torch.sum(y_neg * y_pred_neg) + smooth) / (torch.sum(y_neg) + smooth)
    return true_neg

def tversky(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    true_pos = torch.sum(y_true_f * y_pred_f)
    false_neg = torch.sum(y_true_f * (1 - y_pred_f))
    false_pos = torch.sum((1 - y_true_f) * y_pred_f)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return torch.pow((1 - pt_1), gamma)
