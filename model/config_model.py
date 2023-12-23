from model.mininet import MiniNetv2
from model.linear_regression import LinearRegression

model_dictionary = {
    'mininetv2': MiniNetv2,
    'linear_regression': LinearRegression
}

models_require_image_data = [
    'mininetv2'
]

def load_model_from_config(config):
    model_name = config['model']

    try:
        return model_dictionary[model_name].from_config(config)
    except:
        raise ValueError(f"Unknown model: {model_name}")
    
def model_accepts_image_data(config):
    model_name = config['model']

    if model_name in models_require_image_data:
        return True
    
    return False

    # try:
    #     return (model_dictionary[model_name].from_config(config)).accepts_image_data()
    # except:
    #     raise ValueError(f"Could not get model accepts_image_data state: {model_name}")
    
