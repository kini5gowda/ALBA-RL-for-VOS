import json
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as f1
import torch.optim as optim

from config.persistence import paths
from data_loader import DAVIS2017Loader
from model import SelectionNetwork as MPPN

from PIL import Image

def getbbox(img):
   # the output of the line below is a numpy array
    #img = torch.Tensor.cpu(img).detach().numpy()
    #img = img.astype(bool)
    #rows = np.nonzero(img.any(axis=1))[0]
    #cols = np.img.any(axis=0)
    #cols = np.nonzero(img.any(axis=0))[0]
    #rmin, rmax = np.where(rows)[0][[0, -1]]
    #cmin, cmax = np.where(cols)[0][[0, -1]]
    img = (img > 0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return rmin, rmax, cmin, cmax
    #return rmin,rmax,cmin,cmax


def parse_args():
    device_id = sys.argv[1]  # str
    return 'cuda:' + device_id


def save_model(model, epoch):
    path = paths['supervised']['models.select']
    filename = 'ep{:03d}.pt'.format(epoch)
    torch.save(model.state_dict(), path + filename)


def standardize(x):
    return (x - x.mean()) / x.std()

def class_accuracy(y_true, y_pred):
        c1_true = y_true[y_true == 0]
        c1_pred = y_pred[y_true == 0]
        c2_true = y_true[y_true == 1]
        c2_pred = y_pred[y_true == 1]
        c1_acc = (c1_true == c1_pred).float().mean().item()
        c2_acc = (c2_true == c2_pred).float().mean().item()
        return [c1_acc, c2_acc]


def run_epoch(data_loader, model, criterion, optimizer, ground_truth, mode, device, batch_size=16):
    assert mode in ('train', 'val')
    losses = []
    accuracy = []
    #ff_batch, vf_batch, label_batch = [], [], []
    ff_batch = torch.empty(batch_size, 3, 256, 256).to(device)
    vf_batch = torch.empty(batch_size, 256).to(device)
    label_batch = torch.empty(batch_size).long().to(device)
    batch_count = 0
    positive, negative = 0, 0
    for episode in data_loader:
        mask_proposals = episode.mask_proposals
        optical_flow = episode.optical_flow
        mask_features = episode.mask_features
        #labels = episode.labels
        num_proposals, H, W = mask_proposals.shape
        #num_proposals, H, W = episode.mask_proposals.shape
        optical_flow = standardize(optical_flow.reshape(2,H,W))
        optical_flow = torch.Tensor.cpu(optical_flow).detach().numpy()
        rmin,rmax,cmin,cmax = getbbox(optical_flow)
        optical_flow = optical_flow[rmin:rmax,cmin:cmax]
        optical_flow = np.resize(optical_flow,(2,H,W))


        for i in range(num_proposals):
            try:
                gt_label = ground_truth[episode.seq_id][episode.img_id][i]
            except IndexError:
                break
            proposal = torch.Tensor.cpu(mask_proposals[i]).detach().numpy()
            rmin,rmax,cmin,cmax=getbbox(proposal)
            proposal = proposal[rmin:rmax,cmin:cmax]
            proposal = np.resize(proposal,(H,W))
            proposal = torch.from_numpy(proposal)
            optical_flow_ = torch.from_numpy(optical_flow)
            optical_flow_ = optical_flow_.to(device)
            #print(optical_flow.shape)
            proposal = proposal.to(device)
            ff_batch[batch_count] = f.interpolate(torch.cat([
                proposal.float().unsqueeze(0),
                optical_flow_
            ]).unsqueeze(0), size=(256,256), mode='nearest').squeeze()
            #ff_batch.append(flow_features)
            #vf_batch.append(episode.mask_features[i].mean(2).mean(1))
            vf_batch[batch_count] = mask_features[i].mean(2).mean(1)
            #if labels[0][i] == 0:
              
                 #label_batch[batch_count] = 0
             #   negative += 1
            #else:
                #label_batch[batch_count] = 1
                #positive += 1
            #print(torch.tensor(0))
            if gt_label == 0:
                label_batch[batch_count]=0
            else:
                label_batch[batch_count]=1
            batch_count += 1
            if batch_count == batch_size:
                if mode == 'train':
                    optimizer.zero_grad()

                #ff_batch = torch.stack(ff_batch)
                #vf_batch = torch.stack(vf_batch)
                #label_batch = torch.stack(label_batch).to(device)
                state = {'flow_features': ff_batch, 'visual_features': vf_batch}
                logits = model(state)
                loss = criterion(logits, label_batch)
                losses.append(loss.item())

                #acc = (label_batch == logits.argmax(dim=1)).float().mean()
                #accuracy.append(acc.item())

                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                #ff_batch, vf_batch, label_batch = [], [], []
                #batch_count = 0
                losses.append(loss.item())

                acc = class_accuracy(label_batch, logits.argmax(dim=1))
                accuracy.append(acc)

                #ff_batch = torch.empty(batch_size, 3, 256, 256).to(device)
                #vf_batch = torch.empty(batch_size, 256).to(device)
                #label_batch = torch.empty(batch_size).long().to(device)

                batch_count = 0

    #return np.mean(losses), np.mean(accuracy)
    #total_samples = positive + negative
    #print('[{}]: Total samples: {} | Positive: {} ({}) | Negative: {} ({})'.format(mode, total_samples, positive, positive / total_samples, negative, negative / total_samples))
    return np.mean(losses), tuple(np.nanmean(accuracy, axis=0))


def main():
    device = parse_args()
    model = MPPN()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(device)
    class_weights = torch.tensor([1., 4.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
            # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('Loading training data...')
    train_data_loader = DAVIS2017Loader('train', device)
    print('Loading validation data...')
    val_data_loader = DAVIS2017Loader('val', device)
    with open('gt_train.json', 'r') as fp:
        gt_train = json.load(fp)
    with open('gt_val.json', 'r') as fp:
        gt_val = json.load(fp)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    num_epochs = 80
    print('Training for %d epochs.' % num_epochs)
    for epoch in range(num_epochs):
        ts_start = datetime.now()

        tl, tac = run_epoch(train_data_loader, model, criterion, optimizer, gt_train, 'train', device)
        train_loss.append(tl)
        train_acc.append(tac)

        vl, vac = run_epoch(val_data_loader, model, criterion, optimizer, gt_val, 'val', device)
        val_loss.append(vl)
        val_acc.append(vac)

        ts_end = datetime.now()
        epoch_interval = (ts_end - ts_start).total_seconds()
        print('Epoch %d: [Train loss: %f, Train accuracy: (0) %f (1) %f] '
              '[Val loss: %f, Val accuracy: (0) %f (1) %f] Done in %ds.'
              % (epoch + 1, train_loss[-1], train_acc[-1][0], train_acc[-1][1],
                 val_loss[-1], val_acc[-1][0], val_acc[-1][1], epoch_interval))
        save_model(model, epoch)


if __name__ == '__main__':
    main()
