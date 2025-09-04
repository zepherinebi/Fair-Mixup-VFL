import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        representation = self.fc1_drop(out)
        return representation
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]

class Aggregator(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Aggregator,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc2_drop(out)
        logit = self.fc3(out)
        return torch.sigmoid(logit)
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]