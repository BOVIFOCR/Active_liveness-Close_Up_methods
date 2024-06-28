import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    

    def __init__(self,N=7,distortionFeatureSize=2278):
        # call the parent constructor
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv 1 + pooling 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Conv 2 + pooling 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Conv 1 + pooling 1
        conv1_w = self.__compute_new_size(distortionFeatureSize,5,1,1)
        conv1_h = self.__compute_new_size(N,5,1,1)
        pool1_w = self.__compute_new_size(conv1_w,3,2,1)
        pool1_h = self.__compute_new_size(conv1_h,3,2,1)
        # Conv 2 + pooling 2
        conv2_w = self.__compute_new_size(pool1_w,3,1,1)
        conv2_h = self.__compute_new_size(pool1_h,3,1,1)
        pool2_w = self.__compute_new_size(conv2_w,3,2,1)
        pool2_h = self.__compute_new_size(conv2_h,3,2,1)
        
        # Calculate the size of the fully connected layer input based on N and flattened vector
        self.fc1_input_size = 32 * pool2_h * pool2_w
        
        # Fully connected layers
        self.fully_connected = nn.Sequential(
            # fc1 + drop-out 1
            nn.Linear(self.fc1_input_size, 1024),
            nn.ReLU(),
            # fc2 + drop-out 2
            nn.Linear(1024, 192),
            nn.ReLU(),
            # fc3 + drop-out 3
            nn.Linear(192, 1),
            #nn.Sigmoid()
        )
        
        return


    def __compute_new_size(self,size,kernel_size,stride,padding):
        return (size-kernel_size+2*padding)//stride + 1


    def forward(self, x):
        # Convolution layers
        x = self.conv_layers(x)
        # Reshape for fully connected layers
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = self.fully_connected(x)
        return x
    
    
    def loadWeights(self,weights_path):
        # Load the weights
        pretrained_weights = torch.load(weights_path)

        # Load weights into the model
        self.load_state_dict(pretrained_weights)
        return
    