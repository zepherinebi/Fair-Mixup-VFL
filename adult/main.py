import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from numpy.random import beta
import random

from dataset import preprocess_adult_data
from model import Net, Aggregator
from utils import train_dp, evaluate_dp
from utils import train_eo, evaluate_eo

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.set_device(2)
def run_experiments(method='mixup', mode='dp', lam=0.5, num_exp=10):
    '''
    Retrain each model for 10 times and report the mean ap and dp.
    '''

    ap = []
    gap = []
    hidden_size1=60
    hidden_size2=60
    hidden_size3=60
    for i in range(num_exp):
        print('On experiment', i)
        # get train/test data
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed = i)
    
        X_train1_sample = X_train[:, :40]
        X_train2_sample = X_train[:, 40:80]
        X_train3_sample = X_train[:, 80:121]
        # print(X_train1_sample.shape)
        # print(X_train2_sample.shape)
        # print(X_train3_sample.shape)

        X_val1_sample = X_val[:, :40]
        X_val2_sample = X_val[:, 40:80]
        X_val3_sample = X_val[:, 80:121]
        X_test1_sample = X_test[:, :40]
        X_test2_sample = X_test[:, 40:80]
        X_test3_sample = X_test[:, 80:121]
        # initialize model
        model_1 = Net(input_size=len(X_train1_sample[0]),hidden_size=hidden_size1).cuda()
        model_2 = Net(input_size=len(X_train2_sample[0]),hidden_size=hidden_size2).cuda()
        model_3 = Net(input_size=len(X_train3_sample[0]),hidden_size=hidden_size3).cuda()
        Aggre=Aggregator(input_size=180,hidden_size=180).cuda()
        optimizer = optim.Adam(model_1.get_parameters()+model_2.get_parameters()+model_3.get_parameters()+Aggre.get_parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # run experiments
        ap_val_epoch = []
        gap_val_epoch = []
        ap_test_epoch = []
        gap_test_epoch = []
        for j in tqdm(range(10)):
            if mode == 'dp':
                train_dp(model_1,model_2,model_3,Aggre, criterion, optimizer, X_train1_sample,X_train2_sample,X_train3_sample,A_train, y_train, method, lam)
                ap_val, gap_val = evaluate_dp(model_1,model_2,model_3,Aggre, X_val1_sample,X_val2_sample,X_val3_sample,y_val, A_val)
                ap_test, gap_test = evaluate_dp(model_1,model_2,model_3,Aggre, X_test1_sample, X_test2_sample,X_test3_sample,y_test, A_test)
            elif mode == 'eo':
                train_eo(model_1,model_2,model_3, Aggre,criterion, optimizer, X_train1_sample,X_train2_sample,X_train3_sample, y_train, method, lam)
                ap_val, gap_val = evaluate_eo(model_1,model_2,model_3, Aggre,X_val1_sample,X_val2_sample,X_val3_sample, y_val, A_val)
                ap_test, gap_test = evaluate_eo(model_1,model_2,model_3, Aggre,X_test1_sample,X_test2_sample,X_test3_sample, y_test, A_test)

            if j > 0:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_val_epoch.append(gap_val)
                gap_test_epoch.append(gap_test)
            print(f"After epoch {j}:")
            print(f"Validation AP: {ap_val:.4f}, Validation GAP: {gap_val:.4f}")
            print(f"Test AP: {ap_test:.4f}, Test GAP: {gap_test:.4f}")
        # best model based on validation performance
        idx = gap_val_epoch.index(min(gap_val_epoch))
        gap.append(gap_test_epoch[idx])
        ap.append(ap_test_epoch[idx])


    print('--------AVG---------')
    print('Average Precision', np.mean(ap))
    print(mode + ' gap',  np.mean(gap))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--method', default='mixup', type=str, help='mixup/GapReg/erm')
    parser.add_argument('--mode', default='dp', type=str, help='dp/eo')
    parser.add_argument('--lam', default=0.5, type=float, help='Lambda for regularization')
    args = parser.parse_args()

    run_experiments(args.method, args.mode, args.lam)