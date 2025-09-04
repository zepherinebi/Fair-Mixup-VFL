import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

import random
from torchvision import transforms
from torch.utils.data import Dataset
# torch.cuda.set_device(2)
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
random.seed(1)
class CelebA(Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, gender=None, target=None):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = np.vstack(dataframe.labels.values).astype(float)
        self.gender_id = 20
        label_np = np.vstack(dataframe.labels.values)  # Ensure label_np is 2D
        if label_np.shape[1] == 1:
            label_np = np.squeeze(label_np, axis=1)

        if gender is not None:
            #print('yes')
            gender_idx = np.where(label_np[:, self.gender_id] == gender)[0]
            if target is not None:
                target_idx = np.where(label_np[:, target_id] == target)[0]
                idx = list(set(gender_idx) & set(target_idx))
                self.file_names = self.file_names[idx]
                self.labels = np.vstack(dataframe.labels.values[idx]).astype(float)
            else:
                self.file_names = self.file_names[gender_idx]
                self.labels = np.vstack(dataframe.labels.values[gender_idx]).astype(float)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.labels[index]

        # First, delete the gender_id
        all_labels = np.delete(label, self.gender_id)

        # Update the target_id index after deleting the gender_id
        adjusted_target_id = self.target_id if self.target_id < self.gender_id else self.target_id - 1

        # Then, delete the target_id
        all_labels = np.delete(all_labels, adjusted_target_id)

        # Extract target and gender labels
        gender_label = label[self.gender_id]
        target_label = label[self.target_id]

        if self.transform:
            image = self.transform(image)

        return image, target_label, torch.tensor(all_labels, dtype=torch.float32), gender_label

   

def get_loader(df, data_path, target_id, batch_size, gender=None, target=None):
    dl = CelebA(df, data_path, target_id, transform=tfms, gender=gender, target=target)

    if 'train' in data_path:
        dloader = torch.utils.data.DataLoader(dl, shuffle=True, batch_size=batch_size, num_workers=3, drop_last=True)
    else:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3,drop_last=True)

    return dloader

def evaluate(model_1,model_2, model_linear, dataloader):
    y_scores = []
    y_true = []
    for i, (inputs, target, labels,gender) in enumerate(dataloader):
        inputs, target, labels = inputs.cuda(), target.float().cuda(), labels.float().cuda()
        # print(labels.shape)
        # print(gender.shape)
        feat_1 = model_1(inputs)
        feat_2 = model_2(labels)
        feat= torch.cat((feat_1,feat_2),dim=1)
        pred = model_linear(feat).detach()

        y_scores.append(pred[:, 0].data.cpu().numpy())
        y_true.append(target.data.cpu().numpy())

    y_scores = np.concatenate(y_scores)
    y_true = np.concatenate(y_true)
    ap = average_precision_score(y_true, y_scores)
    return ap, np.mean(y_scores)

def BCELoss(pred, target):
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
