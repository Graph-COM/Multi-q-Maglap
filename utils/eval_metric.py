import torch
from torch.nn.modules import Module

def top_rank_speedup(y_pred, y, config_marker, k=5):
    # output = 2 - (min of y among indices from top 5 y_pred) / (min of y among all indices)
    start_idx = 0
    outputs = []
    for num_configs in config_marker:
        end_idx = start_idx + num_configs
        y_hat = y_pred[start_idx: end_idx] # prediction
        y_gt = y[start_idx: end_idx] # ground truth
        _, indices = torch.sort(y_hat)
        best_top_k = torch.min(y_gt[indices[:k]])
        best = torch.min(y_gt)
        slowdown = best_top_k / best - 1
        outputs.append(1 - slowdown)
    return torch.tensor(outputs).sum()


class L1_loss_per_config(Module):
    def __init__(self, reduction='mean'):
        super(L1_loss_per_config, self).__init__()
        self.loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.reduction = reduction

    def forward(self, y_pred, y, config_marker):
        # final loss = L1 loss per config
        start_idx = 0
        total_loss = 0.
        for num_config in config_marker:
            end_idx = start_idx + num_config
            total_loss += self.loss(y_pred[start_idx:end_idx], y[start_idx:end_idx]) / num_config
            start_idx = end_idx
        if self.reduction == 'sum':
            return total_loss
        elif self.reduction == 'mean':
            return total_loss / len(config_marker) # average per-config loss in the batch


def generate_cross_entropy_mask(y_label, num_classes):
    mask = torch.cumsum(y_label == num_classes - 1, dim=-1) < 2
    return mask



