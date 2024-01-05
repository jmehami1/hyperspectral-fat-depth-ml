import torch.nn as nn
from model.base_regression import RegressionModel

class LinearRegression(RegressionModel):
    """Linear regression model in PyTorch

    Args:
        RegressionModel (class): Regression model base class
    """    
    def __init__(self, input_channels):
        super().__init__(input_channels)
        self.accepts_image_data = False
        self.model_name = "Linear Regression"
        
        self.linear = nn.Linear(input_channels, 1)
        

    def forward(self, x):
        out = self.linear(x)
        return out