import os
import argparse
import pandas as pd
import numpy as np
from numpy.random import beta
from tqdm import tqdm
import pickle
from pprint import pprint

from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch import optim

import itertools
from model import *
from utils import *
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.set_device(1)

def test_model(model_0,model_1, model_linear, dataloader, dataloader_0, dataloader_1,file_to_save):
    model_0.eval()
    model_1.eval()
    model_linear.eval()

    ap, _ = evaluate(model_0,model_1, model_linear, dataloader)
    ap_0, score_0 = evaluate(model_0,model_1,model_linear, dataloader_0)
    ap_1, score_1 = evaluate(model_0,model_1,model_linear, dataloader_1)
    gap = abs(score_0 - score_1)
    result_str = "AP: {:.4f} DP Gap: {:.4f}".format(ap, gap)
    
    # Print to console
    pprint(result_str)
    
    # Write to the file
    file_to_save.write(result_str + '\n')

def process_gender_list(gender):
    # Split indices based on gender
    gender_0 = [i for i, g in enumerate(gender) if g == 0]
    gender_1 = [i for i, g in enumerate(gender) if g == 1]

    # Shuffle the lists
    random.shuffle(gender_0)
    random.shuffle(gender_1)

    half_len = len(gender) // 2
    # Function to randomly select elements from a list
    def select_random_elements(lst, count):
        selected = set()
        while len(selected) < count:
            selected.update(random.sample(lst, min(count - len(selected), len(lst))))
        return list(selected)

    # Adjusting lists based on their lengths
    if len(gender_0) >= half_len:
        gender_0 = select_random_elements(gender_0, half_len)
    else:
        extra_indices = select_random_elements(gender_0, half_len - len(gender_0))
        gender_0.extend(extra_indices)

    if len(gender_1) >= half_len:
        gender_1 = select_random_elements(gender_1, half_len)
    else:
        extra_indices = select_random_elements(gender_1, half_len - len(gender_1))
        gender_1.extend(extra_indices)
    return gender_0, gender_1

def find_matching_indices(idx_0, idx_1, gender_0, gender_1):
    # Ensure all lists are of the same length
    min_length = min(len(idx_0), len(idx_1))
    gender_0 = set(gender_0[:min_length])
    gender_1 = set(gender_1[:min_length])
    # Find matching indices
    idxx = [i for i in range(min_length) if idx_0[i] in gender_0 and idx_1[i] in gender_1]
    return idxx

def filter_mismatched_gender_indices(idx_0, idx_1, gender_tensor):
    """
    Returns a list of indices where the gender label for idx_0 is 0 and the gender label for idx_1 is not 0.

    :param idx_0: List of indices from the first group.
    :param idx_1: List of indices from the second group.
    :param gender_tensor: Tensor containing gender labels.
    :return: List of indices with mismatched gender.
    """
    # Extract gender labels using the given indices
    gender_0 = gender_tensor[idx_0]
    gender_1 = gender_tensor[idx_1]

    # Create a mask for mismatched genders where gender_0 is 0 and gender_1 is not 0
    mask = (gender_0 == 0) & (gender_1 != 0)

    # Convert the mask to a list of indices
    mismatched_indices = [i for i, match in enumerate(mask) if match]
    return mismatched_indices

def fit_model_1(epochs, dataloader, dataloader_0, dataloader_1, mode='mixup_smooth', lam=100,gamma_threshold=1):
    pprint("Epoch: {}".format(epochs))

    len_dataloader = len(dataloader)
    # len_dataloader = min(len(dataloader), len(dataloader_0), len(dataloader_1))
    len_dataloader = int(len_dataloader)
    data_iter = iter(dataloader)
    data_iter_0 = iter(dataloader_0)
    data_iter_1 = iter(dataloader_1)

    model_0.train()  
    model_1.train()  
    model_linear.train()

    for it in range(len_dataloader):

        inputs,target,labels,gender=next(data_iter)
        inputs, target,labels,gender = inputs.cuda(), target.float().cuda(), labels.float().cuda(), gender.float().cuda()
 
        feat0 = model_0(inputs)
        feat1 = model_1(labels)
        feat = torch.cat((feat0,feat1),dim=1)
        ops = model_linear(feat)
        loss_sup = criterion(ops[:,0], target)
        

        if mode == 'GapReg':
            feat00 = model_0(inputs_0)
            feat01 = model_1(target_0)
            feat = torch.cat((feat0,feat1),dim=1) 
            ops_0 = model_linear(feat)
            feat = model(inputs_1)
            ops_1 = model_linear(feat)

            loss_gap = torch.abs(ops_0.mean() - ops_1.mean())
            loss = loss_sup + lam*loss_gap

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} | Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

        elif mode == 'mixup':
            alpha = 1
            gamma = beta(alpha, alpha)

            # Input Mixup
            inputs_mix = inputs_0 * gamma + inputs_1 * (1 - gamma)
            inputs_mix = inputs_mix.requires_grad_(True)
            feat = model_0(inputs_mix)
            ops = model_linear(feat).sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())
            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup {:.7f}".format(loss_sup, loss_grad))

        elif mode == 'mixup_manifold':
            alpha = 1
            gamma = beta(alpha, alpha)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(feat.detach().cpu().numpy())

            # kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(feat.cpu())
            A_train = kmeans.labels_
            # idx_0 = np.where(A_train == 0)[0].tolist()
            # idx_1 = np.where(A_train == 1)[0].tolist()

            # Indices where A_train is 0
            idx_0 = np.where(A_train == 0)[0].tolist()
            
            # Indices where A_train is 1
            idx_1 = np.where(A_train == 1)[0].tolist()
            
            # Check lengths and balance
            if len(idx_0) > len(idx_1):
                extra_indices = np.random.choice(idx_1, size=len(idx_0) - len(idx_1), replace=True).tolist()
                idx_1.extend(extra_indices)
            elif len(idx_1) > len(idx_0):
                extra_indices = np.random.choice(idx_0, size=len(idx_1) - len(idx_0), replace=True).tolist()
                idx_0.extend(extra_indices)

            original_pairs = set(zip(idx_0, idx_1))

            # Generate new pairs and avoid duplicates
            new_pairs = set()
            for i in idx_1:
                while True:
                    j = random.choice(idx_0)
                    new_pair = (i, j)
                    if new_pair not in original_pairs and new_pair not in new_pairs:
                        new_pairs.add(new_pair)
                        break

            # Add the new pairs to the original pairs   
            for pair in new_pairs:
                idx_0.append(pair[0])
                idx_1.append(pair[1])

            feat0_0=feat0[idx_0]
            feat0_1=feat0[idx_1]
            feat1_0=feat1[idx_0]
            feat1_1=feat1[idx_1]
            # Manifold Mixup
            feat_0 = torch.cat((feat0_0,feat1_0),dim=1)
            feat_1 = torch.cat((feat0_1,feat1_1),dim=1)
            inputs_mix_0 = feat0_0 * gamma + feat0_1 * (1 - gamma)
            inputs_mix_1 = feat1_0 * gamma + feat1_1 * (1 - gamma)
            inputs_mix_0 = inputs_mix_0.requires_grad_(True)
            inputs_mix_1 = inputs_mix_1.requires_grad_(True)
            inputs_mix = torch.cat((inputs_mix_0,inputs_mix_1),dim=1)
            x_d = (feat_1 - feat_0).view(inputs_mix.shape[0], -1)
            mask = filter_mismatched_gender_indices(idx_0, idx_1, gender)
            if len(mask)<64:
                extra_mask = np.random.choice(mask, size=(64 - len(mask)), replace=True).tolist()
                mask.extend(extra_mask)
            else:
                mask=mask[:64]
            inputs_mix = inputs_mix[mask]
            
            x_d = x_d[mask] 
            ops = model_linear(inputs_mix).sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())
            loss = loss_sup + lam * loss_grad
            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup Manifold {:.7f}".format(loss_sup, loss_grad))

        else:
            loss = loss_sup
            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f}".format(loss_sup))

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_linear.zero_grad()
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        optimizer_linear.step()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CelebA Experiment')
    parser.add_argument('--method', default='mixup', type=str, help='mixup/mixup_manifold/GapReg/erm')
    parser.add_argument('--lam', default=20, type=float, help='Lambda for regularization')
    parser.add_argument('--target_id', default=2, type=int, help='2:attractive/31:smile/33:wavy hair')
    parser.add_argument('--gamma_threshold', default=1, type=float, help='gamma is range in (gamma_threshold,1)')
    args = parser.parse_args()

    # Load Celeb dataset
    target_id = args.target_id
    with open('/data/bxt_data/data_frame.pickle', 'rb') as handle:
        df = pickle.load(handle)
    train_df = df['train']
    valid_df = df['val']
    test_df = df['test']

    train_dataloader = get_loader(train_df, '/data/bxt_data/tmp/train/', target_id, 128)
    valid_dataloader = get_loader(valid_df, '/data/bxt_data/tmp/val/', target_id, 128)
    test_dataloader = get_loader(test_df, '/data/bxt_data/tmp/test/', target_id, 128)

    train_dataloader_0 = get_loader(train_df, '/data/bxt_data/tmp/train/', target_id, 64, gender = '0')
    train_dataloader_1 = get_loader(train_df, '/data/bxt_data/tmp/train/', target_id, 64, gender = '1')
    valid_dataloader_0 = get_loader(valid_df, '/data/bxt_data/tmp/val/', target_id, 64, gender = '0')
    valid_dataloader_1 = get_loader(valid_df, '/data/bxt_data/tmp/val/', target_id, 64, gender = '1')
    test_dataloader_0 = get_loader(test_df, '/data/bxt_data/tmp/test/', target_id, 64, gender = '0')
    test_dataloader_1 = get_loader(test_df, '/data/bxt_data/tmp/test/', target_id, 64, gender = '1')

    # model
    model_0 = ResNet18_Encoder(pretrained=False).cuda()

    model_1 = FairnessInsensitiveFeaturePlatform2(input_dim=38).cuda()
    
    model_linear = LinearModel().cuda()

    criterion = nn.BCELoss()
    optimizer_1 = optim.Adam(model_0.parameters(), lr = 1e-3)
    optimizer_2 = optim.Adam(model_1.parameters(), lr = 1e-3)
    
    #optimizer = optim.Adam(model_1.parameters(), lr = 1e-3)
    optimizer_linear = optim.Adam(model_linear.parameters(), lr = 1e-3)
    base_filename = f"id{args.target_id}_lam{args.lam}_gamma{args.gamma_threshold}_result_{args.method}"
    filename = base_filename + ".txt"
    file_counter = 1

        # Check if the file exists and add a counter to the filename
    while os.path.isfile(filename):
        filename = f"{base_filename}({file_counter}).txt"
        file_counter += 1
    for i in range(1, 100):
        with open(filename, 'a') as result_file:
            fit_model_1(i, train_dataloader, train_dataloader_0, train_dataloader_1, args.method, args.lam, args.gamma_threshold)
            print('val:')
            test_model(model_0,model_1, model_linear, valid_dataloader, valid_dataloader_0, valid_dataloader_1,result_file)
            print('test:')
            test_model(model_0,model_1, model_linear, test_dataloader, test_dataloader_0, test_dataloader_1,result_file)
       

    