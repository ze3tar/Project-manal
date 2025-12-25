import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, target):
    return F.cross_entropy(logits, target.long())


def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    dims = tuple(range(1, target.ndim))
    intersection = torch.sum(probs * target, dims)
    union = torch.sum(probs + target, dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_fruit_loss(logits, target, lambda_ce=1.0, lambda_dice=1.0):
    ce = cross_entropy_loss(logits, target)
    dice = dice_loss(logits, target)
    return lambda_ce * ce + lambda_dice * dice
