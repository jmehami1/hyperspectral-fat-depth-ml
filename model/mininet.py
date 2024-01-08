import torch.nn as nn
import torch.nn.functional as F
from model.base_regression import RegressionModel

class DephwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution

    1. It performs a spatial convolution independently for each input channel
    2. It performs a pointwise (1x1) convolution onto the output channels
    """
    def __init__(self, input_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, dilation, input_channels, bias)
        self.pointwise = nn.Conv2d(input_channels, out_channels, 1, 1, 0, 1, 1, bias)
        self.bn = nn.BatchNorm2d(input_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pointwise(x)
        return x

class MultiDilationDephwiseSeparableConv2d(nn.Module):
    """
    Multi-dilation depthwise separable convolution

    It performs two parallel depthwise convolutions, one with dilation rate 1 and another
    with dilation rate >= 1. Then, their outputs are added and a pointwise convolution
    is applied.
    """
    def __init__(self, input_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding,       1 , input_channels, bias)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, dilation, input_channels, bias)
        self.pointwise = nn.Conv2d(input_channels, out_channels, 1, 1, 0, 1, 1, bias)
        self.bn1 = nn.BatchNorm2d(input_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(input_channels, eps=1e-3)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        output = x1 + x2
        output = self.pointwise(output)
        return output


class MiniNetv2Downsample(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.conv = MultiDilationDephwiseSeparableConv2d(input_channels, out_channels, 3, stride=2, padding=0, dilation=1)

    def forward(self, x):
        output = self.conv(x)
        # print(output.shape)
        return output

class MiniNetv2Module(nn.Module):
    def __init__(self, input_channels, out_channels, dilation):
        super().__init__()
        self.conv = MultiDilationDephwiseSeparableConv2d(input_channels, out_channels, 3, stride=1, padding='same', dilation=dilation)

    def forward(self, x):
        output = self.conv(x)
        # print(output.shape)
        return output

class MiniNetv2Upsample(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_channels, out_channels, 3, stride=2, padding=0, dilation=1)

    def forward(self, x):
        output = self.conv(x)
        # print(output.shape)
        return output

class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x
    
def add_padding_even_dimension(image):
    """Add padding of 1 pixel thickness to original image height or width if either are even. Padding will be added to either bottom or right of image.

    Args:
        image (Image): original image

    Returns:
        Image: Padded image
    """
    height, width = image.size(-2), image.size(-1)

    # Check if height is even, pad bottom side
    if height % 2 == 0:
        image = F.pad(image, (0, 0, 0, 1), mode='constant', value=0)
        # image = F.pad(image, (0, 0, 1, 1), mode='constant', value=0)
        

    # Check if width is even, pad right side
    if width % 2 == 0:
        image = F.pad(image, (0, 1, 0, 0), mode='constant', value=0)
        # image = F.pad(image, (1, 1, 0, 0), mode='constant', value=0)

    return image


class MiniNetv2(RegressionModel):
    """MiniNetv2 CNN regression model in PyTorch

    Args:
        RegressionModel (class): Regression model base class
    """ 
    def __init__(self, input_channels):
        super().__init__(input_channels)
        self.accepts_image_data = True
        self.model_name = "MiniNetv2"

        # 1. Downsample block
        self.d1 = MiniNetv2Downsample(input_channels, 16)
        self.d2 = MiniNetv2Downsample(16, 64)
        self.m_downsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(10)])
        self.d3 = MiniNetv2Downsample(64, 128)

        # 2. Feature extractor block
        rates = [1, 2, 1, 4, 1, 8, 1, 16, 1, 1, 1, 2, 1, 4, 1, 8]
        self.m_feature = nn.ModuleList([MiniNetv2Module(128, 128, rate) for rate in rates])

        # 3. Refinement block
        self.d4 = MiniNetv2Downsample(input_channels, 16)
        self.d5 = MiniNetv2Downsample(16, 64)

        # 4. Upsample block
        self.up1 = MiniNetv2Upsample(128, 64)
        self.m_upsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(4)])
        self.up2 = MiniNetv2Upsample(64, 16)
        self.output = MiniNetv2Upsample(16, 1)

    def forward(self, x):
        # pad (if necessary)
        x_padded = add_padding_even_dimension(x)

        # Refinement
        d4 = self.d4(x_padded)
        d5 = self.d5(d4)

        # Downsample
        d1 = self.d1(x_padded)
        d2 = self.d2(d1)
        m_downsample = d2
        for m in self.m_downsample:
            m_downsample = m(m_downsample)
        d3 = self.d3(m_downsample)

        # Feature
        m_feature = d3
        for m in self.m_feature:
            m_feature = m(m_feature)


        # Upsample
        up1 = self.up1(m_feature)
        m_upsample = up1 + d5
        for m in self.m_upsample:
            m_upsample = m(m_upsample)
        m_upsample = self.up2(m_upsample)
        output = self.output(m_upsample)

        output = Interpolate(size=x.shape[2:])(output)

        return output
    
    # @classmethod
    # def from_config(model_class, config):
    #     try:
    #         input_channels = config['input_channels']
    #         out_channels = config['output_channels']
    #         instance = model_class(input_channels, out_channels)
    #         return instance
    #     except:
    #         raise ValueError(f"config missing member variables for MiniNet-v2")
        
    # def accepts_image_data(self):
    #     return self.accepts_image_data
