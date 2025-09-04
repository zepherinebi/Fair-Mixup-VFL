import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#torch.cuda.set_device(2)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x):
        outputs = self.resnet(x)
        outputs = outputs.view(-1,512,8,8)
        x = self.avg(outputs).view(-1, 512)
        return x

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(544, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        #print(x.shape)
        #x = self.avg(x).view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return torch.sigmoid(outputs)
class FairnessModel(nn.Module):
    def __init__(self, image_input_shape, feature_input_shape):
        super(FairnessModel, self).__init__()

        # Image processing layers
        self.conv1 = nn.Conv2d(image_input_shape[0], 32, 5) # Assuming the first dimension is channels
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(32, 128, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)

        # Calculate the output shape after convolutional layers
        # (This might need to be adjusted based on the actual input shape)
        # self.flattened_dim = 128 * ((image_input_shape[1]-8)//2) * ((image_input_shape[2]-8)//2)
        self.flattened_dim = 476288
        
        self.fc1 = nn.Linear(self.flattened_dim, 512)

        # Fairness-insensitive feature platform
        self.fc_feature1 = nn.Linear(feature_input_shape, 32)
        self.fc_feature2 = nn.Linear(32, 32)
        self.fc_feature3 = nn.Linear(32, 32)

        # Aggregation model
        self.fc_aggregate = nn.Linear(512+32, 512) # Assuming concatenation of image_rep (512) and feature_rep (32)

    def forward(self, image_input, feature_input):
        # Image processing
        x = self.dropout1(self.pool1(F.relu(self.conv1(image_input))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))
        #print(x.shape)

        x = x.view(-1, self.flattened_dim)
        image_rep = F.relu(self.fc1(x))

        # Fairness-insensitive feature platform
        feature_rep = F.relu(self.fc_feature1(feature_input))
        feature_rep = F.relu(self.fc_feature2(feature_rep))
        feature_rep = self.fc_feature3(feature_rep)


        return image_rep, feature_rep

class FairnessInsensitiveFeaturePlatform2(nn.Module):
    def __init__(self, input_dim):
        super(FairnessInsensitiveFeaturePlatform2, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        
    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # print(x.shape)
        return x


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Raw data representation
        self.raw_data_rep = nn.Linear(in_features=512,out_features=512)  # NOTE: replace <input_size> with the actual input size
        
        # Target task classification
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=512, out_features=512)
        
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        
        self.dropout3 = nn.Dropout(0.2)
        self.logit = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.raw_data_rep(x)
        
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        
        x = self.dropout3(x)
        x = F.softmax(self.logit(x), dim=1)  # softmax along the appropriate dimension
        
        return x

