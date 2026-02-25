import torch
import torch.nn.functional as F

# Class weights: background=1.0, fruit=10.0 (apples are ~2-5% of pixels)
FRUIT_CLASS_WEIGHTS = torch.tensor([1.0, 10.0])


def cross_entropy_loss(logits, target):
    weight = FRUIT_CLASS_WEIGHTS.to(logits.device)
    return F.cross_entropy(logits, target.long(), weight=weight)


def dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    dims = tuple(range(1, target.ndim))
    intersection = torch.sum(probs * target, dims)
    union = torch.sum(probs + target, dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_fruit_loss(logits, target, lambda_ce=1.0, lambda_dice=1.0, use_class_weights=True):
    if use_class_weights:
        ce = cross_entropy_loss(logits, target)
    else:
        ce = F.cross_entropy(logits, target.long())
    dice = dice_loss(logits, target)
    return lambda_ce * ce + lambda_dice * dice


def multi_view_consistency_loss(all_view_logits):
    """KL-divergence consistency loss across multiple views.

    Args:
        all_view_logits: list of [B, 2, H, W] logits, one per view.

    Returns:
        Scalar loss (mean KL across views relative to the mean prediction).
    """
    if len(all_view_logits) < 2:
        return torch.tensor(0.0, device=all_view_logits[0].device)

    # Compute mean softmax probability across views as pseudo-target
    probs = [F.softmax(logit, dim=1) for logit in all_view_logits]
    mean_prob = torch.stack(probs, dim=0).mean(dim=0)  # [B, 2, H, W]
    mean_log = (mean_prob + 1e-8).log()

    total_kl = torch.tensor(0.0, device=all_view_logits[0].device)
    for prob in probs:
        log_prob = (prob + 1e-8).log()
        # KL(view || mean) = sum(view * log(view / mean))
        kl = F.kl_div(mean_log, prob, reduction="batchmean", log_target=False)
        total_kl = total_kl + kl

    return total_kl / len(probs)
