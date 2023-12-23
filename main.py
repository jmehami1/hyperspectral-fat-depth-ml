import os
import torch
from utils.utils import has_folder, find_max_batch_size, delete_files_folder
from data.dataset import load_fat_depth_regression, FatDepthDataset, ToTensor, split_training_data_of_samples
from data.scaling import DataScaler
from data.dimensionality_reduction import DimensionReducer
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.data import random_split

from model.config_model import model_accepts_image_data

from model.mininet import MiniNetv2

import torch.nn as nn
import torch.optim as optim

import time
import matplotlib.pyplot as plt

from model.config_model import load_model_from_config
from train.trainer import train_best_config
import multiprocessing


from utils.config import load_tune_config, load_main_config, save_config_file,load_config_file
from train.tuner import start_hyperparameter_tuning, start_preprocessing_tuning, json_config_to_run_config, tune_preprocessing, tune_hyperparameters
from data.preprocessing import preprocessing_from_config



def plot_reflectance_false_color(cube, false_color_channels, title):
    scaler = DataScaler()
    scaler.fit(cube)
    cube = scaler.transform(cube)
    img = cube[:, :, false_color_channels]
    plt.imshow(img)
    plt.show(block=False)
    axs = plt.gca()
    axs.set_title(title)


if __name__ == '__main__':

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    delete_files_folder(os.path.join(os.getcwd(), "raytune"))

    # parameters from config file
    config = load_main_config()
    split_ratio_training_testing = config["split_ratio_training_testing"]  
    split_ratio_training_validation = config["split_ratio_training_validation"]
    path_training = config["path_training"] 
    path_testing = config["path_testing"]
    result_dir = config["result_dir"]
    num_epochs = config["number_epochs"]
    perform_tune_hyperparameters = config["tune_hyperparameters"]
    project_name = config["project_name"]
    experiment_name = config["experiment_name"]

    training_testing_is_same_dir = (path_training == path_testing)

    # get data directory names
    # training_name = os.path.basename(path_training)

    # if training_testing_is_same_dir:
    #     testing_name = training_name
    # else:
    #     testing_name = os.path.basename(path_testing)

    # check if data directories exist 
    if not has_folder(path_training, "fat_depth_regression"):
        raise Exception(f"{path_training} does not have the fat_depth_regression folder")
    else:
        path_training = os.path.join(path_training, "fat_depth_regression")
    
    if not has_folder(path_testing, "fat_depth_regression"):
        raise Exception(f"{path_testing} does not have the fat_depth_regression folder")
    else:
        path_testing = os.path.join(path_testing, "fat_depth_regression")

    # make directory for training and testing sets
    result_dir = os.path.join(result_dir, project_name, experiment_name)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # for reproducible results
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    num_processes = multiprocessing.cpu_count()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load fat depth regression data as dictionaries (not using pytorch datasets here)
    if training_testing_is_same_dir:
        fat_depth_data = load_fat_depth_regression(path_training)
        fat_depth_training, fat_depth_testing = split_training_data_of_samples(fat_depth_data, split_ratio_training_testing)
    else:
        fat_depth_training = load_fat_depth_regression(path_training)
        fat_depth_testing = load_fat_depth_regression(path_testing)

    # copy of all training and validation data used to train best model
    fat_depth_training_all = fat_depth_training

    # create validation regression data dictionary
    fat_depth_training, fat_depth_validation = split_training_data_of_samples(fat_depth_training, split_ratio_training_validation)

    input_data_size_worst = fat_depth_training["reflectance_cube"].shape[1:]
    input_labels_size_worst = fat_depth_training["fat_depth_map"].shape[1:]

    # get current gpu max batch size
    temp_config = {
        "reduction_method": "PCA",
        "scaler_method": "normalize",
        "reduction_components": 20,
        "model": "mininetv2",
        "learning_rate": 0.005,
        "batch_size": 32,
        "momentum": 0.0,
        "input_channels": 20,
        "output_channels": 1
    }
    temp_model = load_model_from_config(temp_config)
    max_batch_sizes = find_max_batch_size(temp_model, device, input_data_size_worst, input_labels_size_worst)

    # Transforms to training dataset
    apply_transform = transforms.Compose([
        ToTensor()
    ])

    # dataset_training = FatDepthDataset(fat_depth_validation, scaler, reducer, transform=apply_transform, load_as_image=True)

    
    # # data loaders
    # train_data_loader = DataLoader(dataset_training, batch_size=1)  
    # data, labels = next(iter(train_data_loader))

    if perform_tune_hyperparameters:
        tune_config = load_tune_config()
        perform_tune_preprocessing = config["preprocessing"]["tune"]

        best_preprocessing_json = os.path.join(result_dir, "best_preprocessing.json")
        best_hyperparameter_json = os.path.join(result_dir, "best_hyperparameter.json")
        # tune preprocessing with linear regression
        if perform_tune_preprocessing:
            print("Tuning data preprocessing with linear regression")

            # temp_config = {
            #     "reduction_method": "PCA",
            #     "scaler_method": "normalize",
            #     "reduction_components": 20,
            #     "model": "linear_regression",
            #     # "model": "mininetv2",
            #     "learning_rate": 0.005,
            #     "batch_size": 1000,
            #     "momentum": 0.0,
            # }

            # tune_preprocessing(temp_config, fat_depth_training=fat_depth_training, fat_depth_validation=fat_depth_validation, 
                                    # apply_transform=apply_transform, device=device, max_batch_sizes=max_batch_sizes, num_epochs=num_epochs)
            
            preprocessing_resuts = start_preprocessing_tuning(tune_config["preprocessing"], project_name, fat_depth_training, fat_depth_validation, apply_transform,\
                                device, max_batch_sizes, num_samples=1, cpu_per_trial=5, num_epochs=5)
            
            best_result = preprocessing_resuts.get_best_result("validation_loss", "min")
            save_config_file(best_result.config, best_preprocessing_json)

        if os.path.exists(best_preprocessing_json):
            preprocessing_config = load_config_file(best_preprocessing_json)
        else:
            preprocessing_config = config["preprocessing"]

        dataset_training, dataset_validation = preprocessing_from_config(preprocessing_config, fat_depth_training, fat_depth_validation, apply_transform, True)

        # temp_config = {
        #         "reduction_method": "PCA",
        #         "scaler_method": "normalize",
        #         "reduction_components": 20,
        #         "model": "mininetv2",
        #         "learning_rate": 0.005,
        #         "batch_size": 1000,
        #         "momentum": 0.0,
        # }

        # tune_hyperparameters(temp_config, dataset_training, dataset_validation, device, max_batch_sizes, num_epochs=50)


        hyperparameter_resuts = start_hyperparameter_tuning(tune_config["search_space"], project_name, dataset_training, dataset_validation, device, max_batch_sizes,\
                                                                 num_epochs=50, num_samples=20, cpu_per_trial=num_processes/2, gpu_per_trial=0.5)

        best_result = hyperparameter_resuts.get_best_result("validation_loss", "min")
        save_config_file(best_result.config, best_hyperparameter_json)   

        
        # tune hyperparameters

    # temp_config = {
    #         "reduction_method": "PCA",
    #         "scaler_method": "normalize",
    #         "reduction_components": 20,
    #         "model": "mininetv2",
    #         "learning_rate": 0.005,
    #         "batch_size": 4,
    #         "momentum": 0.0,
    #         "input_channels": 20,
    #         "output_channels": 1
    # }

    # train_best_config(temp_config, dataset_training, device, num_epochs=50)


#     # data scaler and dimensionality reduction objects
#     scaler = DataScaler(method=scaler_method)
#     reducer = DimensionReducer(n_components=reduction_components, reduction_method=reduction_method)

#     # Transforms to training dataset
#     apply_transform = transforms.Compose([
#         ToTensor()
#     ])

#     # create pytorch datasets 
#     # dataset_training = FatDepthDataset(fat_depth_training, scaler, reducer, transform=apply_transform, load_as_image=True)
#     # dataset_validation = FatDepthDataset(fat_depth_validation, scaler, reducer, transform=apply_transform, load_as_image=True, testing=True)
#     # dataset_testing = FatDepthDataset(fat_depth_testing, scaler, reducer, transform=apply_transform, testing=True, load_as_image=True)

#     # first_data = dataset_training[0]
#     # data, labels = first_data
#     # print(type(data), type(labels))
#     # print(data.shape, labels.shape)

#     temp_best_config = {
#         'model': 'mininetv2',
#         'batch_size': batch_size,
#         'learning_rate': learning_rate,
#         'momentum': momentum,
#         'input_channels': reduction_components,
#         'output_channels': 1
#     }

#     dataset_training = FatDepthDataset(fat_depth_training_all, scaler, reducer, transform=apply_transform, load_as_image=True)

#     train_best_config(temp_best_config, dataset_training, device, num_epochs=50)


#     # first_data = dataset_testing[0]
#     # data, labels = first_data
#     # print(type(data), type(labels))

#     # data loaders
#     train_data_loader = DataLoader(dataset_training, batch_size=batch_size)  
#     test_data_loader = DataLoader(dataset_testing)

#     data, labels = next(iter(train_data_loader))

#     # model
#     model = MiniNetv2(reduction_components, 1).to(device)
#     summary(model=model, input_size=data.shape)

#     # loss and optimizer
#     criterion = nn.MSELoss(reduction='mean')
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

#     print("starting training...")

#     for epoch in range(1, num_epochs):
#         model.train()
#         batch_loss = []
#         start_time = time.time()

#         for i, (inputs, labels) in enumerate(train_data_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             inputs, labels = inputs.float(), labels.float()
#             labels = torch.squeeze(labels)
#             optimizer.zero_grad()
#             output = torch.squeeze(model(inputs))
#             loss = criterion(output,labels)
#             loss.backward()
#             optimizer.step()
#             batch_loss.append(loss.item())
        
#         train_loss = np.mean(batch_loss)
        
#         print("\tEpoch: {:>2}/{:} \t Average Training Loss: {:>3.3f} \t Epoch Time: {:>5.3f}s".format(epoch + 1, num_epochs, train_loss, (time.time() - start_time)))
    


    
    
#     print("end of script")



# # import numpy as np
# # import torch
# # from my_models import MLPRegressor
# # import os
# # import pandas as pd
# # import torchvision.transforms as transforms



# # import torch.nn as nn
# # from torch.nn import functional
# # import torch.optim as optim
# # from torchinfo import summary
# # from torch.utils.data import Dataset, random_split
# # import time
# # from ray import tune, air
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler
# import multiprocessing
# import matplotlib.pyplot as plt
# # from sklearn.metrics import r2_score
# from torchmetrics.regression import R2Score


# import wandb
# from ray.air.integrations.wandb import setup_wandb
# from ray.air.integrations.wandb import WandbLoggerCallback


# def train_ray_tune(config, dataset, train_validate_ratio, input_layer, output_layer, device, num_epochs=50):
# 	batch_size = int(config["batch_size"])
# 	learning_rate = config["learning_rate"]
# 	layer1 = int(config["layer1"])
# 	layer2 = int(config["layer2"])
# 	momentum = config["momentum"]

# 	# model
# 	hidden_layers = [layer1, layer2]
# 	model = MLPRegressor(input_layer, output_layer, hidden_layers).to(device)

# 	#loss and optimizer
# 	criterion = nn.MSELoss(reduction='mean')
# 	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 	# ray tune checkpoint
# 	# checkpoint = session.get_checkpoint()

# 	# checkpoints seems to add significant overhead when saving models.
# 	# if checkpoint:
# 	# 	checkpoint_state = checkpoint.to_dict()
# 	# 	start_epoch = checkpoint_state["epoch"]
# 	# 	model.load_state_dict(checkpoint_state["model_state_dict"])
# 	# 	optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
# 	# else:
# 	# 	start_epoch = 0

# 	start_epoch = 0

# 	# data
# 	train_size = int(len(dataset) * train_validate_ratio)
# 	validate_size = len(dataset) - train_size
# 	dataset_train, dataset_validation = random_split(dataset, [train_size, validate_size])

# 	train_loader = torch.utils.data.DataLoader(
#         dataset_train, batch_size=batch_size, shuffle=True
#     )
	
# 	validation_loader = torch.utils.data.DataLoader(
#         dataset_validation
#     )


# 	# ray tune training and validation loop
# 	for epoch in range(start_epoch, num_epochs):
# 		model.train()
# 		batch_loss = []

# 		for i, (inputs, labels) in enumerate(train_loader):
# 			optimizer.zero_grad()
# 			output = model(inputs)
# 			loss = criterion(output,labels)
# 			loss.backward()
# 			optimizer.step()
# 			batch_loss.append(loss.item())

# 		train_loss = np.mean(batch_loss)

# 		model.eval()
# 		validation_loss = []
# 		with torch.no_grad():
# 			for i, (inputs, labels) in enumerate(validation_loader):
# 				output = model(inputs)
# 				loss = criterion(output,labels)
# 				validation_loss.append(loss.item())

# 		validation_loss = np.mean(validation_loss)

# 		# checkpoint_data = {
#         #     "epoch": epoch,
#         #     "model_state_dict": model.state_dict(),
#         #     "optimizer_state_dict": optimizer.state_dict(),
# 	    # 	"validation_loss": validation_loss
#         # }

# 		# checkpoint = Checkpoint.from_dict(checkpoint_data)
# 		session.report({
# 			"validation_loss": validation_loss,
# 			"training_loss": train_loss},
#             # checkpoint=checkpoint,
#         )

# def train_best_config(config, dataset_train, dataset_test, input_layer, output_layer, device, num_epochs=10):
# 	# parameters to tune
# 	batch_size = int(config["batch_size"])
# 	learning_rate = config["learning_rate"]
# 	layer1 = int(config["layer1"])
# 	layer2 = int(config["layer2"])
# 	momentum = config["momentum"]

# 	# model
# 	hidden_layers = [layer1, layer2]
# 	model = MLPRegressor(input_layer, output_layer, hidden_layers).to(device)

# 	#loss and optimizer
# 	criterion = nn.MSELoss(reduction='mean')
# 	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# 	# data
# 	# train_size = int(len(dataset) * train_test_ratio)
# 	# test_size = len(dataset) - train_size
# 	# dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

# 	train_loader = torch.utils.data.DataLoader(
#         dataset_train, batch_size=batch_size, shuffle=True
#     )
	
# 	test_loader = torch.utils.data.DataLoader(
#         dataset_test
#     )

# 	# training
# 	model.train()
# 	for epoch in range(num_epochs):
# 		batch_loss = []
# 		start_time = time.time()

# 		for i, (inputs, labels) in enumerate(train_loader):
# 			optimizer.zero_grad()
# 			output = model(inputs)
# 			loss = criterion(output,labels)
# 			loss.backward()
# 			optimizer.step()
# 			batch_loss.append(loss.item())

# 		train_loss = np.mean(batch_loss)
# 		wandb.log({"Training loss": train_loss})
# 		print("\tEpoch: {:>2}/{:} \t Average Training Loss: {:>3.3f} \t Epoch Time: {:>5.3f}s".format(epoch + 1, num_epochs, train_loss, (time.time() - start_time)))

# 	# testing
# 	model.eval()
# 	with torch.no_grad():
# 		test_loss = []
# 		y_true = []
# 		y_predicted = []

# 		for inputs, labels in test_loader:
# 			output = model(inputs)
# 			loss = criterion(output,labels)
# 			test_loss.append(loss.item())

# 			y_predicted.append(output)
# 			y_true.append(labels)

# 		r2score = R2Score()
# 		r2_test = r2score(y_predicted, y_true)

# 	test_loss = np.mean(test_loss)
# 	wandb.run.summary["R2 Test"] = r2_test.numpy()
# 	wandb.run.summary["Test Loss"] = test_loss
# 	print("R2 test: {0:2.3f}, test loss: {1:2.3e}".format(r2_test, test_loss))


# class StandardizeTransform(object):
#     def __init__(self, dim):
#         self.dim = dim

#     def __call__(self, tensor):
#         mean = torch.mean(tensor, dim=self.dim, keepdim=True)
#         std = torch.std(tensor, dim=self.dim, keepdim=True)
#         return (tensor - mean) / std
	
	
# class RealEstateDataset(Dataset):
# 	def __init__(self, device):
# 		csv_file = os.path.join(os.curdir, "data", "real_estate_valuation_data_set.csv")
# 		dataset = pd.read_csv(csv_file, delimiter=",")

# 		dataset.hist()
# 		plt.suptitle("Raw Data")
# 		plt.show(block=False)

# 		data_transform = StandardizeTransform(dim=0)

# 		dataset_tensor = torch.from_numpy(dataset.to_numpy(dtype=np.float32)).to(device)
# 		X = dataset_tensor[:, 0:6]
# 		self.y = dataset_tensor[:, -1]
# 		self.X = data_transform(X)
# 		# self.X = functional.normalize(X, dim=0)
# 		# self.X = functional.std (X, dim=0)
		

# 		dataset_normalized = pd.DataFrame(torch.cat((self.X, self.y.reshape(-1,1)), 1).cpu().numpy(), columns=dataset.columns)

# 		dataset_normalized.hist()
# 		plt.suptitle("Normalized Data")
# 		plt.show(block=False)

# 		self.num_samples = dataset.shape[0]


# 	def __getitem__(self, index):
# 		return self.X[index,:], self.y[index]
	
# 	def __len__(self):
# 		return self.num_samples
	


# if __name__ == "__main__":
# 	# device and pc details
# 	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
# 	if device.type == 'cuda':
# 		num_gpu = torch.cuda.device_count()
# 	else:
# 		num_gpu = 0

# 	num_cores = multiprocessing.cpu_count()
# 	print("Training device: {0}".format(device))
# 	print("Number of CPU cores: {0:2d}".format(num_cores))
# 	print("Number of GPUs : {0:2d}".format(num_gpu))

# 	project_name = "real_estate"

# 	wandb.init(mode="disabled")
	    
# 	#dataset 
# 	dataset = RealEstateDataset(device)
# 	num_samples = len(dataset)
# 	first_row = dataset[0]
# 	features, labels = first_row
# 	print(features, labels)

# 	# parameters for problem
# 	input_layer = features.shape[0]
# 	output_layer = 1
# 	train_test_split_ratio = 0.8
# 	train_validate_split_ratio = 0.8

# 	# split data train (train and validate) and test
# 	train_size = int(len(dataset) * train_test_split_ratio)
# 	test_size = len(dataset) - train_size
# 	dataset_train, dataset_test = random_split(dataset, [train_size, test_size])

# 	config = {
#     "layer1": tune.grid_search(range(input_layer, input_layer*2, 2)),
#     "layer2": tune.grid_search(range(input_layer, input_layer*2, 2)),
#     "learning_rate": tune.grid_search([10**x for x in np.arange(-4, 0, 0.5)]),
#     "batch_size": tune.grid_search([1, 2, 4, 20, 50, 100]),
#     "momentum": tune.grid_search(np.linspace(0, 1, 6)),
#     "wandb": {"project": project_name},
# 	}

# 	# config = {
#     # "layer1": tune.grid_search(range(input_layer, input_layer*2, 10)),
#     # "layer2": tune.grid_search(range(input_layer, input_layer*2, 10)),
#     # "learning_rate": tune.grid_search([10**x for x in range(-4, 0, 3)]),
#     # "batch_size": tune.grid_search([1]),
#     # "momentum": tune.grid_search(np.linspace(0, 1, 1)),
# 	# "wandb": {"project": project_name},
# 	# }

# 	# scheduler used to terminate bad trials early, pause trials, and change hyperparameters
# 	scheduler = ASHAScheduler(
#         metric="validation_loss",
#         mode="min",
#         max_t=20,
#         grace_period=1,
#         reduction_factor=2,
#     )

# 	# launches hyperparater tuning jobs with given tuning function, scheduler and parameter search space
# 	tuner = tune.Tuner(
#         tune.with_resources(
#             tune.with_parameters(train_ray_tune, dataset=dataset_train, 
# 				 train_validate_ratio=train_validate_split_ratio, input_layer=input_layer, output_layer=output_layer, device=device),
#             resources={"cpu": 1, "gpu": 1/10}
#         ),
#         tune_config=tune.TuneConfig(
#             scheduler=scheduler,
#             num_samples=10,
#         ),
#         param_space=config,
# 		run_config=air.RunConfig(
# 			verbose=1, 
# 			name=project_name, 
# 			local_dir="./raytune",
# 			checkpoint_config=air.CheckpointConfig(
# 				num_to_keep=1,
# 				# checkpoint_score_attribute="validation_loss",
# 				# checkpoint_score_order="min"
# 				),
# 			callbacks=[
#                 WandbLoggerCallback(project=project_name)
#             ]
# 		)
#     )

# 	results = tuner.fit()
# 	best_result = results.get_best_result("validation_loss", "min")
# 	print("Best trial config: {}".format(best_result.config))
# 	print("Best trial: training loss {0:2.3e}, validation loss: {1:2.3e}".format(best_result.metrics["training_loss"], best_result.metrics["validation_loss"]))

# 	wandb.init(
# 		project=project_name,
# 		name="mlp_standardized_features",
# 		config=best_result.config)

# 	train_best_config(config=best_result.config, dataset_train=dataset_train, dataset_test=dataset_test, input_layer=input_layer, output_layer=output_layer, device=device, num_epochs=200)


