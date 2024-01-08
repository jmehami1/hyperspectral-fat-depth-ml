# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_regression import RegressionModel


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        output = torch.cat([self.conv(x), self.pool(x)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, x):

        output = self.conv3x1_1(x)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+x)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 32)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(32,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, out_channels, 1, stride=1, padding=0, bias=True)

    def forward(self, x, predict=False):
        output = self.initial_block(x)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,32))
        self.layers.append(non_bottleneck_1d(32, 0, 1))
        self.layers.append(non_bottleneck_1d(32, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 32, out_channels, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        return output

# class Interpolate(nn.Module):
#     def __init__(self, size, mode='bilinear'):
#         super().__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode

#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
#         return x


#ERFNet
class ERFNet(RegressionModel):
    """ERFNet CNN regression model in PyTorch

    Args:
        RegressionModel (class): Regression model base class
    """ 
    def __init__(self, input_channels, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__(input_channels)
        self.accepts_image_data = True
        self.model_name = "ERFNet"

        if (encoder == None):
            self.encoder = Encoder(input_channels, 1)
        else:
            self.encoder = encoder

        self.decoder = Decoder(1)

    def forward(self, x, only_encode=False):
        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output = self.encoder(x)    #predict=False by default
            output = self.decoder.forward(output)
            return output