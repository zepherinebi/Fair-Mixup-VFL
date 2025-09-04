import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.set_device(2)
torch.autograd.set_detect_anomaly(True)

def sample_batch_sen_idx(X, A, y, batch_size, s):    
    batch_idx = np.random.choice(np.where(A==s)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = batch_x.clone().detach().cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

def sample_batch_sen(X, A, y, batch_size, s):    
    batch_idx = np.random.choice(len(X), size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = batch_x.clone().detach().cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()
    batch_A = A[batch_idx]
    batch_A = torch.tensor(batch_A).cuda().float()
    return batch_x, batch_y, batch_A

def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    batch_idx = []
    for i in range(2):
        idx = list(set(np.where(A==s)[0]) & set(np.where(y==i)[0]))
        batch_idx += np.random.choice(idx, size=batch_size, replace=False).tolist()

    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = batch_x.clone().detach().cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y

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
def train_dp(model_1,model_2,model_3,Aggregator,criterion, optimizer, X_train_1,X_train_2,X_train_3,gender, y_train, method, lam, batch_size=500, niter=100):
    model_1.train()
    model_2.train()
    model_3.train()
    Aggregator.train()
    X_train_1=torch.tensor(X_train_1).cuda().float()
    X_train_2=torch.tensor(X_train_2).cuda().float()
    X_train_3=torch.tensor(X_train_3).cuda().float()
    y_train=torch.tensor(y_train).cuda().float()
    gender=torch.tensor(gender).cuda().float()
    
    for it in range(niter):
        
         
        batch_idx = np.random.choice(len(y_train), size=batch_size, replace=False).tolist()
        batch_X_train_1 = X_train_1[batch_idx]
        batch_X_train_2 = X_train_2[batch_idx]
        batch_X_train_3 = X_train_3[batch_idx]
        
        batch_y = y_train[batch_idx]
        batch_gender = gender[batch_idx]
        
        representation_1=model_1(batch_X_train_1)
        representation_2=model_2(batch_X_train_2)
        representation_3=model_3(batch_X_train_3)

        unified_representation=torch.cat((representation_1,representation_2,representation_3),dim=1)
    
        output = Aggregator(unified_representation)
        # print(output.size())
        
        loss_sup = criterion(output, batch_y)
        # X_train_1 = X_train_1.clone().detach().cuda().float()
        # X_train_2 = X_train_2.clone().detach().cuda().float()
        # X_train_3 = X_train_3.clone().detach().cuda().float()

        

        if method == 'mixup':
            # Fair Mixup
            alpha = 1
            gamma = beta(alpha, alpha)
            
            unified_representation_np = unified_representation.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(unified_representation_np)

            A_train = kmeans.labels_
            
            idx_0 = np.where(A_train == 0)[0].tolist()
            
            idx_1 = np.where(A_train == 1)[0].tolist()
           
            if len(idx_0) > len(idx_1):
                extra_indices = np.random.choice(idx_1, size=len(idx_0) - len(idx_1), replace=True).tolist()
                idx_1.extend(extra_indices)
            elif len(idx_1) > len(idx_0):
                extra_indices = np.random.choice(idx_0, size=len(idx_1) - len(idx_0), replace=True).tolist()
                idx_0.extend(extra_indices)

            original_pairs = set(zip(idx_0, idx_1))

            new_pairs = set()
            for i in idx_1:
                while True:
                    # Randomly select an element from idx_0
                    j = random.choice(idx_0)

                    # Check for duplicates. If not duplicates, add to new_pairs
                    new_pair = (i, j)
                    if new_pair not in original_pairs and new_pair not in new_pairs:
                        new_pairs.add(new_pair)
                        break
                    
            # add the new pairs to the original pairs
            for pair in new_pairs:
                idx_0.append(pair[0])
                idx_1.append(pair[1])
            
            batch_x_0 = unified_representation[idx_0] 
            batch_x_1 = unified_representation[idx_1] 
            
            batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)
            batch_x_d = batch_x_1 - batch_x_0
            
            mask = filter_mismatched_gender_indices(idx_0, idx_1, batch_gender)
           
            mix_len = batch_size//2
            if len(mask)< mix_len:
                extra_mask = np.random.choice(mask, size=(mix_len - len(mask)), replace=True).tolist()
                mask.extend(extra_mask)
            else:
                mask=mask[:mix_len]
            #print(mask)
            batch_x_mix = batch_x_mix[mask]
            
            batch_x_d = batch_x_d[mask] 
            output = Aggregator(batch_x_mix)
            
            # gradient regularization
            gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]

            
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = torch.abs(E_grad)

        else:
            # ERM
            loss_reg = 0

        

        # final loss
        loss = loss_sup + lam*loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





def evaluate_dp(model_1,model_2,model_3,Aggregator, X_test1,X_test2,X_test3, y_test, A_test):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    Aggregator.eval()
    # calculate DP gap
    idx_0 = np.where(A_test==0)[0]
    idx_1 = np.where(A_test==1)[0]

    X_test1_0 = X_test1[idx_0]
    X_test2_0 = X_test2[idx_0]
    X_test3_0 = X_test3[idx_0]
    X_test1_1 = X_test1[idx_1]
    X_test2_1 = X_test2[idx_1]
    X_test3_1 = X_test3[idx_1]
    X_test1_0 = torch.tensor(X_test1_0).cuda().float()
    X_test2_0 = torch.tensor(X_test2_0).cuda().float()
    X_test3_0 = torch.tensor(X_test3_0).cuda().float()
    X_test1_1 = torch.tensor(X_test1_1).cuda().float()
    X_test2_1 = torch.tensor(X_test2_1).cuda().float()
    X_test3_1 = torch.tensor(X_test3_1).cuda().float()

    pred1_0 = model_1(X_test1_0)
    pred2_0 = model_2(X_test2_0)
    pred3_0 = model_3(X_test3_0)
    pred_0=Aggregator(torch.cat((pred1_0,pred2_0,pred3_0),dim=1))
    pred1_1 = model_1(X_test1_1)
    pred2_1 = model_2(X_test2_1)
    pred3_1 = model_3(X_test3_1)
    pred_1=Aggregator(torch.cat((pred1_1,pred2_1,pred3_1),dim=1))
    gap = pred_0.mean() - pred_1.mean()
    gap = abs(gap.data.cpu().numpy())

    # calculate average precision
    X_test1_cuda = torch.tensor(X_test1).cuda().float()
    X_test2_cuda = torch.tensor(X_test2).cuda().float()
    X_test3_cuda = torch.tensor(X_test3).cuda().float()
    output_1=model_1(X_test1_cuda)
    output_2=model_2(X_test2_cuda)
    output_3=model_3(X_test3_cuda)
    output = Aggregator(torch.cat((output_1,output_2,output_3),dim=1))
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, gap

