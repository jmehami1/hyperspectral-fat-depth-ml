from model.mininet import MiniNetv2
from model.linear_regression import LinearRegression
from model.erfnet import ERFNet

models_available = {
    'MiniNetv2': MiniNetv2,
    'LinearRegression': LinearRegression,
    'ERFNet': ERFNet
}

def load_model(model_name, input_channels):
    """Load regression model from name

    Args:
        model_name (str): name of model
        input_channels (int): number of input channel features

    Raises:
        ValueError: Model is not available

    Returns:
        model: An pytorch regression model object
    """    
    try:
        return (models_available[model_name])(input_channels)
    except:
        raise ValueError(f"Unknown model: {model_name}")
    
def model_accepts_image_data(model_name):
    """Get the type of data that is accepted by model. True if model accepts full images else False for list of pixels

    Args:
        model_name (str): name of model

    Raises:
        ValueError: Could not get accepted image data of model

    Returns:
        bool: True accepts images, else False for list of pixels
    """    
    try:
        return (models_available[model_name](1)).accepts_image_data()
    except:
        raise ValueError(f"Could not get accepted image data type of {model_name}")
    
