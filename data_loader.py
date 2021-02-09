import os

import numpy as np
import torch
import sys

# Soft links.
DAVIS_2017_PATH = 'DAVIS-2017'
DAVIS_2017_FLOW_PATH = 'DAVIS-2017-flow'
DAVIS_2017_MASK_PATH = 'DAVIS-2017-detectron-masks'
DAVIS_2017_FEATURE_PATH = 'DAVIS-2017-detectron-mask-features'
DAVIS_2017_GROUND_TRUTH_PATH = 'DAVIS-2017-ground-truth'



class _FrameData(object):
    def __init__(self, seq_id, img_id, is_initial, mask_proposals, optical_flow, mask_features, annotation):
        self.seq_id = seq_id
        self.img_id = img_id
        self.is_initial = is_initial
        self.mask_proposals = mask_proposals
        self.optical_flow = optical_flow
        self.mask_features = mask_features
        self.annotation = annotation

    def __repr__(self):
        return '{}:{}'.format(self.seq_id, self.img_id)

class DAVIS2017Loader(object):
    def __init__(self, subset, device):
        self.subset = subset
        if subset not in ('train', 'val'):
            print('Invalid subset: "%s". Defaulting to "train".' % subset)
            self.subset = 'train'
        self.device = device
        self.blacklist = set([
            # No proposals generated.
            'surf:00053', 'surf:00054'
        ])
        self.seq_ids, self.img_ids = self._get_ids()
        self._n = len(self.img_ids)
        self.proposal_paths, self.flow_paths, self.feature_paths, self.annotation_paths = self._get_pre_computed_data()
        self.is_initial = [(img_id == '00000') for img_id in self.img_ids]

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        img_id = self.img_ids[idx]
        is_initial = self.is_initial[idx]
        mask_proposals = torch.load(self.proposal_paths[idx]).to(self.device)
        optical_flow = torch.from_numpy(read_flow_file(self.flow_paths[idx])).to(self.device)
        mask_features = torch.load(self.feature_paths[idx]).to(self.device)
        annotation = torch.from_numpy(torch.load(self.annotation_paths[idx])).to(self.device)
        return _FrameData(seq_id, img_id, is_initial, mask_proposals, optical_flow, mask_features, annotation)

    def _get_ids(self):
        p = os.path.join(DAVIS_2017_PATH, 'ImageSets/2017', self.subset + '.txt')
        with open(p, 'r') as fp:
            lines = fp.read().strip().split('\n')
        seq_ids, img_ids = [], []
        for line in lines:
            seq_id = line.strip()
            img_path = os.path.join(DAVIS_2017_PATH, 'JPEGImages', '480p', seq_id)
            img_filenames = sorted([x for x in os.listdir(img_path) if x.lower().endswith('.jpg')])
            anno_path = os.path.join(DAVIS_2017_PATH, 'Annotations_unsupervised', '480p', seq_id)
            anno_filenames = sorted([x for x in os.listdir(anno_path) if x.lower().endswith('.png')])
            assert len(img_filenames) == len(anno_filenames)
            for i in range(len(img_filenames)):
                img_id = img_filenames[i].split('.')[0]
                assert img_id == anno_filenames[i].split('.')[0]
                if '{}:{}'.format(seq_id, img_id) in self.blacklist:
                    continue
                seq_ids.append(seq_id)
                img_ids.append(img_id)
        return seq_ids, img_ids

    def _get_pre_computed_data(self):
        proposal_paths, flow_paths, feature_paths, annotation_paths = [], [], [], []
        for i in range(self._n):
            seq_id = self.seq_ids[i]
            img_id = self.img_ids[i]
            proposal_paths.append(os.path.join(DAVIS_2017_MASK_PATH, self.subset, seq_id, img_id + '.pt'))
            flow_paths.append(os.path.join(DAVIS_2017_FLOW_PATH, seq_id, img_id + '.flo'))
            feature_paths.append(os.path.join(DAVIS_2017_FEATURE_PATH, self.subset, seq_id, img_id + '.pt'))
            annotation_paths.append(os.path.join(DAVIS_2017_GROUND_TRUTH_PATH, self.subset, seq_id, img_id + '.pt'))
        return proposal_paths, flow_paths, feature_paths, annotation_paths


def read_flow_file(filename):
    f = open(filename, 'rb')
    tag = np.fromfile(f, np.float32, count=1)[0]
    assert tag == 202021.25
    W = np.fromfile(f, np.int32, count=1)[0]
    H = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*W*H)
    flow = np.resize(data, (int(H), int(W), 2))
    f.close()
    return flow
