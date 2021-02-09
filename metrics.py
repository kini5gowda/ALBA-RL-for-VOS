import torch
from scipy.optimize import linear_sum_assignment


def IoU(m1, m2):
    intersection = (m1 * m2).sum().float()
    union = torch.clamp((m1 + m2), 0, 1).sum().float()
    if union > 0.:
        return intersection / union
    return torch.tensor(0.)


def IoS(m1, m2):
    """ Intersection over self. """
    intersection = (m1 * m2).sum().float()
    s = m1.sum().float()
    if s > 0.:
        return intersection / s
    return torch.tensor(0.)


def IoU_multi_obj(m1, m2):
    n = m1.shape[0]
    m = m2.shape[0]
    iou_matrix = torch.zeros(n, m)
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = IoU(m1[i], m2[j])
    r, c = linear_sum_assignment(-iou_matrix)
    return iou_matrix[r, c].mean()
