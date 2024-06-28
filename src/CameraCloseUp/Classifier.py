import torch
import torch.nn as nn

class Classifier(nn.Module):
    
    def __init__(self,N=24,distortionFeatureSize=2088):
        # call the parent constructor
        super().__init__()

        # Custom kernel
        custom_kernel = torch.tensor(
            [[1., 0., -1.],
            [1., 0., -1.],
            [1., 0., -1.]], dtype=torch.float32)
        custom_kernel = custom_kernel.view(1, 1, 3, 3)
        # Initialize the Conv2d layer with the custom kernel
        self.fixed_conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        # Set the custom weights and disable gradients
        with torch.no_grad():
            self.fixed_conv_layer.weight = nn.Parameter(custom_kernel, requires_grad=False)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Custom layer
            self.fixed_conv_layer,
            # Conv 1 + pooling 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Conv 2 + pooling 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Conv 3 + pooling 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Custom layer
        conv0_w = self.__compute_new_size(distortionFeatureSize,3,1,1)
        conv0_h = self.__compute_new_size(N,3,1,1)
        # Conv 1 + pooling 1
        conv1_w = self.__compute_new_size(conv0_w,5,1,1)
        conv1_h = self.__compute_new_size(conv0_h,5,1,1)
        pool1_w = self.__compute_new_size(conv1_w,3,2,1)
        pool1_h = self.__compute_new_size(conv1_h,3,2,1)
        # Conv 2 + pooling 2
        conv2_w = self.__compute_new_size(pool1_w,3,1,1)
        conv2_h = self.__compute_new_size(pool1_h,3,1,1)
        pool2_w = self.__compute_new_size(conv2_w,3,2,1)
        pool2_h = self.__compute_new_size(conv2_h,3,2,1)
        # Conv 3 + pooling 3
        conv3_w = self.__compute_new_size(pool2_w,3,1,1)
        conv3_h = self.__compute_new_size(pool2_h,3,1,1)
        pool3_w = self.__compute_new_size(conv3_w,3,2,1)
        pool3_h = self.__compute_new_size(conv3_h,3,2,1)

        # Calculate the size of the fully connected layer input based on k and f
        self.fc1_input_size = 64 * pool3_h * pool3_w
        
        # Fully connected layers
        self.fully_connected = nn.Sequential(
            # fc1 + drop-out 1
            nn.Linear(self.fc1_input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # fc2 + drop-out 2
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # fc3 + drop-out 3
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output layer
            nn.Linear(256, 1),
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
        # Zully connected layers
        x = self.fully_connected(x)
        return x
    
    
    def loadWeights(self,weights_path):
        # Load the weights
        pretrained_weights = torch.load(weights_path)

        # Load weights into the model
        self.load_state_dict(pretrained_weights)
        return