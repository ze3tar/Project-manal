# losses.py - Complete with true focal loss
import torch
import torch.nn.functional as F

def focal_loss_with_prob_volume(prob_volume, depth_gt, depth_values, mask, gamma=0):
    """
    True focal loss as per MT-MVSNet paper Equation 7
    
    Args:
        prob_volume: [B, D, H, W] probability distribution over depths
        depth_gt: [B, H, W] ground truth depth
        depth_values: [B, D] depth hypotheses
        mask: [B, H, W] valid pixel mask
        gamma: focal loss gamma (0 for DTU per paper)
    Returns:
        scalar focal loss
    """
    B, D, H, W = prob_volume.shape
    
    # Find nearest depth index for each pixel
    depth_gt_expanded = depth_gt.unsqueeze(1)  # [B, 1, H, W]
    depth_values_expanded = depth_values.view(B, -1, 1, 1)  # [B, D, 1, 1]
    
    # Calculate distance to each depth hypothesis
    depth_diff = torch.abs(depth_values_expanded - depth_gt_expanded)
    gt_index = depth_diff.argmin(dim=1)  # [B, H, W]
    
    # Convert to one-hot
    gt_onehot = F.one_hot(gt_index, num_classes=D).permute(0, 3, 1, 2).float()  # [B, D, H, W]
    
    # Clamp probabilities
    prob_volume = prob_volume.clamp(1e-6, 1.0 - 1e-6)
    
    # Extract probability at ground truth depth
    P_d = (prob_volume * gt_onehot).sum(dim=1)  # [B, H, W]
    
    # Focal loss: -(1 - P_d)^γ * log(P_d)
    focal_weight = (1 - P_d) ** gamma
    loss = -focal_weight * torch.log(P_d)
    
    # Apply mask
    mask = mask.float()
    loss = loss * mask
    
    return loss.sum() / (mask.sum() + 1e-6)


def l1_depth_loss(pred_depth, gt_depth, mask):
    """L1 depth loss (backup/alternative)"""
    if pred_depth.dim() == 3: pred_depth = pred_depth.unsqueeze(1)
    if gt_depth.dim() == 3:   gt_depth   = gt_depth.unsqueeze(1)
    if mask.dim() == 3:       mask       = mask.unsqueeze(1)
    denom = mask.sum().clamp_min(1.0)
    return (F.l1_loss(pred_depth * mask, gt_depth * mask, reduction='sum') / denom)


def cosine_normal_loss(pred_normals, gt_normals, mask, eps=1e-6):
    """Cosine normal loss"""
    if mask.dim() == 3: mask = mask.unsqueeze(1)
    pred_n = pred_normals / (pred_normals.norm(dim=1, keepdim=True) + eps)
    gt_n   = gt_normals   / (gt_normals.norm(dim=1, keepdim=True) + eps)
    cos = (pred_n * gt_n).sum(dim=1, keepdim=True).clamp(-1, 1)
    loss = (1.0 - cos) * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def depth_normal_consistency(depth, normals, K, eps=1e-6):
    """Depth-normal consistency loss"""
    B, _, H, W = depth.shape
    device = depth.device
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    ones = torch.ones_like(x)
    pix = torch.stack([x, y, ones], dim=0).float().unsqueeze(0).repeat(B,1,1,1)
    Kinv = torch.inverse(K).view(B,3,3,1,1)
    rays = (Kinv @ pix.view(B,3,1,H,W)).view(B,3,H,W)
    Z = depth
    X = rays[:,0:1] * Z
    Y = rays[:,1:2] * Z
    Tx = torch.zeros(B,3,H,W, device=device); Ty = torch.zeros_like(Tx)
    Tx[:,:,:,1:] = torch.stack([X[:,:,:,1:]-X[:,:,:,:-1],
                                Y[:,:,:,1:]-Y[:,:,:,:-1],
                                Z[:,:,:,1:]-Z[:,:,:,:-1]], dim=1).squeeze(1)
    Ty[:,:,1:,:] = torch.stack([X[:,:,1:,:]-X[:,:,:-1,:],
                                Y[:,:,1:,:]-Y[:,:,:-1,:],
                                Z[:,:,1:,:]-Z[:,:,:-1,:]], dim=1).squeeze(1)
    nx = Tx[:,1]*Ty[:,2] - Tx[:,2]*Ty[:,1]
    ny = Tx[:,2]*Ty[:,0] - Tx[:,0]*Ty[:,2]
    nz = Tx[:,0]*Ty[:,1] - Tx[:,1]*Ty[:,0]
    n_geom = torch.stack([nx, ny, nz], dim=1)
    n_geom = n_geom / (n_geom.norm(dim=1, keepdim=True) + eps)
    pred_n = normals / (normals.norm(dim=1, keepdim=True) + eps)
    cos = (pred_n * n_geom).sum(dim=1, keepdim=True).clamp(-1, 1)
    return (1.0 - cos).mean()


def edge_aware_smoothness(depth, image, w=1.0):
    """Edge-aware smoothness loss"""
    dx_d = depth[:,:,:,1:] - depth[:,:,:,:-1]
    dy_d = depth[:,:,1:,:] - depth[:,:,:-1,:]
    gray = 0.2989*image[:,0:1] + 0.5870*image[:,1:2] + 0.1140*image[:,2:3]
    dx_i = gray[:,:,:,1:] - gray[:,:,:,:-1]
    dy_i = gray[:,:,1:,:] - gray[:,:,:-1,:]
    wx = torch.exp(-w * dx_i.abs()); wy = torch.exp(-w * dy_i.abs())
    return (wx * dx_d.abs()).mean() + (wy * dy_d.abs()).mean()


