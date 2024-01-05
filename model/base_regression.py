import torch.nn as nn

class RegressionModel(nn.Module):
    """Base class for a regression model in PyTorch

    Args:
        nn (Module): Inherits neural network module from PyTorch
    """    
    def __init__(self, input_channels):
        super().__init__()
        self.accepts_image_data = False
        self.input_channels = input_channels
        self.model_name = "Base"

    def forward(self, x):
        # The forward method should be implemented in the child class
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def requires_image_data(self):
        """Does model require image data or list of pixels

        Returns:
            bool: true if model requires image data, otherwise false for list of pixels
        """        
        return self.accepts_image_data
    
    # @classmethod
    # def from_config(cls, config):
    #     """Load an instance of models given 

    #     Args:
    #         model_class (_type_): _description_
    #         config (_type_): _description_

    #     Raises:
    #         ValueError: _description_

    #     Returns:
    #         _type_: _description_
    #     """        
    #     try:
    #         in_channels = config['input_channels']
    #         instance = cls(in_channels)
    #         return instance
    #     except:
    #         raise ValueError(f"Could not create instance of {cls.model_name} regression model.")
