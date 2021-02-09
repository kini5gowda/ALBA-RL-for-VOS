import collections
import json
import sys

import torch

from data_loader import DAVIS2017Loader
from metrics import IoU


def parse_args():
    subset = sys.argv[1]  # str
    if subset not in ('train', 'val'):
        print('Invalid subset "%s", defaulting to "train".')
        subset = 'train'
    device_id = sys.argv[2]  # str
    return subset, 'cuda:' + device_id


def main(subset, device):
    gt = collections.defaultdict(dict)
    data_loader = DAVIS2017Loader(subset, device)
    for e in data_loader:
        if e.img_id == '00000':
            print(e.seq_id)

        num_proposals, H, W = e.mask_proposals.shape
        num_annotations, H_, W_ = e.annotation.shape
        assert (H == H_) and (W == W_)
        gt[e.seq_id][e.img_id] = [0] * num_proposals
        m = torch.zeros(num_annotations, H, W).to(device)

        for i in range(num_proposals):
            iou_vector = torch.zeros(num_annotations)
            for j in range(num_annotations):
                iou_vector[j] = IoU(e.mask_proposals[i], e.annotation[j])
            if iou_vector.max() < 0.2:
                continue
            max_iou_idx = iou_vector.argmax().item()
            hypothesis_mask = torch.clamp(m[max_iou_idx] + e.mask_proposals[i], 0, 1)
            hypothesis_iou = IoU(hypothesis_mask, e.annotation[max_iou_idx])
            current_iou = IoU(m[max_iou_idx], e.annotation[max_iou_idx])
            if hypothesis_iou >= current_iou:
                m[max_iou_idx] = torch.clamp(m[max_iou_idx] + e.mask_proposals[i], 0, 1)
                gt[e.seq_id][e.img_id][i] = int(max_iou_idx + 1)

    with open('gt_{}.json'.format(subset), 'w') as fp:
        json.dump(gt, fp)
    print('Done!')


if __name__ == '__main__':
    subset, device = parse_args()
    main(subset, device)
