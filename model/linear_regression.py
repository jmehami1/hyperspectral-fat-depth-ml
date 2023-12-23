import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.accepts_image_data = False
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    @classmethod
    def from_config(model_class, config):
        try:
            in_channels = config['input_channels']
            out_channels = config['output_channels']
            instance = model_class(in_channels, out_channels)
            return instance
        except:
            raise ValueError(f"config missing member variables for MiniNet-v2")
        
    def accepts_image_data(self):
        return self.accepts_image_data