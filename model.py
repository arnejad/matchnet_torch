import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

B = 8192

class classifier_net(torch.nn.Module):

    def __init__(self):
        super(classifier_net, self).__init__()
        self.fc1 = torch.nn.Linear(B, 512)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 2)
        # self.sm = torch.nn.Softmax()

    def forward(self, x):

        # print("inp: ", x.shape)
        x = F.relu(self.fc1(x))
        # print("FC1: ", x.shape)
        x = F.relu(self.fc2(x))
        # print("FC2: ", x.shape)
        x = self.fc3(x)
        # print("FC3: ", x.shape)
        x = F.log_softmax(x, -1)
        # print("SM: ", x.shape)
        return x



class feature_net(torch.nn.Module):

    def __init__(self):
        super(feature_net, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv0 = torch.nn.Conv2d(1, 24, 7, padding=3)
        self.conv1 = torch.nn.Conv2d(24, 64, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 96, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(96, 96, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(96, 64, 3, padding=1)
        # an affine operation: y = Wx + b

    def forward(self, x):
        
        # print("inp: ", x.shape)
        # in_ = (x.astype(np.float32) - 128) / 160
        # Max pooling over a (2, 2) window
        x = self.conv0(x)
        # print("conv0: ", x.shape)
        x = F.max_pool2d(F.relu(x), 3, stride=2, padding=1) # Pool0
        # print("pool0: ", x.shape)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv1(x)), 3, stride=2, padding=1) # Pool1
        # print("pool1: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2: ", x.shape)
        x = F.relu(self.conv3(x))
        # print("conv3: ", x.shape)
        x = F.max_pool2d(F.relu(self.conv4(x)), 3, stride=2, padding=1) # Pool4 (num based on paper, which should be the 2nd I beloeve)
        # print("pool4: ", x.shape)
        # x = x.view(-1, self.num_flat_features(x)) #Bottleneck
        # x = x.reshape((len(in_), -1, 1, 1))
        x = torch.flatten(x,  start_dim=1)
        
        return x


class matchnet(torch.nn.Module):

    def __init__(self):
        super(matchnet, self).__init__()

        self.feature1 = feature_net()
        self.feature2 = feature_net()
        self.classifier = classifier_net()
        

    def forward(self, x):
        #split data.
        x1 = x[:,:,:, 0:64]  
        x2 = x[:,:,:, 64:128]  
        
        feat1 = self.feature1(x1)
        feat2 = self.feature2(x2)
        res = self.classifier(torch.cat((feat1, feat2), 1))
        return res

