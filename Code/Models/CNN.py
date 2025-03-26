import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    '''
    CNN. Processes input (bs, 1, d, d) through sequence of conv layers.
    '''

    def __init__(self):
        super(CNN, self).__init__()

        # conv parameters: input channels, output_channels (number of filters), kernel size (square)
        self.conv1 = nn.Conv2d(1, 32, 3, dtype=torch.double)
        self.conv2 = nn.Conv2d(32, 64, 3, dtype=torch.double)
        self.conv3 = nn.Conv2d(64, 64, 3, dtype=torch.double)

        # maxpool parameters: square filter size
        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
          


    def forward(self, x):

        # conv, relu, maxpool
        x = self.conv1(x) # shape = (bs, 32, 26, 26) say d=28
        x = F.relu(x)
        x = self.mp1(x) # shape = (bs, 32, 13, 13)
        
        # conv, relu, maxpool
        x = self.conv2(x) # shape = (bs, 64, 11, 11)
        x = F.relu(x)
        x = self.mp2(x) # shape = (bs, 64, 5, 5)

        # conv, relu, flatten
        x = self.conv3(x) # shape = (bs, 64, 3, 3)
        x = F.relu(x)
        x = torch.flatten(x, 1) # shape = (bs, 576) 576 = 64*3*3

        return x
