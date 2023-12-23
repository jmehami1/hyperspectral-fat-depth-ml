from data.dimensionality_reduction import DimensionReducer
from data.scaling import DataScaler
from data.dataset import FatDepthDataset

def preprocessing_from_config(config, fat_depth_training, fat_depth_testing, apply_transform, load_as_image=True):
    scaler_method = config["scaler_method"]
    reduction_method = config["reduction_method"]
    reduction_components = config["reduction_components"]

     # data scaler and dimensionality reduction objects
    scaler = DataScaler(method=scaler_method)
    reducer = DimensionReducer(n_components=reduction_components, reduction_method=reduction_method)

    dataset_training = FatDepthDataset(fat_depth_training, scaler, reducer, transform=apply_transform, load_as_image=load_as_image)
    dataset_testing = FatDepthDataset(fat_depth_testing, scaler, reducer, transform=apply_transform, load_as_image=load_as_image, testing=True)

    return dataset_training, dataset_testing