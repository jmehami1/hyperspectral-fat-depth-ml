import torch
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from model.config_model import load_model_from_config, model_accepts_image_data
from torch.utils.data import DataLoader

from ray import tune, air
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

from train.trainer import train_epoch, valiate_epoch
from data.preprocessing import preprocessing_from_config

from ray.tune.schedulers import ASHAScheduler
from ray import tune, air

from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

tune_search_space_dict = {
    "choice": tune.choice,
    "uniform": tune.uniform,
    "grid_search": tune.grid_search
}


def json_config_to_run_config(config, project_name):
    run_config = {}

    for parameter in config:
        tune_type = config[parameter]["type"]
        tune_value = config[parameter]["value"]

        if tune_type in ["uniform"]:
            run_config[parameter] = (tune_search_space_dict[tune_type])(tune_value[0], tune_value[1])
        else:
            run_config[parameter] = (tune_search_space_dict[tune_type])(tune_value)

    run_config["wandb"] = { 
        "project": project_name, 
        }

    return run_config

def start_preprocessing_tuning(tune_config, project_name, fat_depth_training, fat_depth_validation, apply_transform,\
                                device, max_batch_sizes, num_epochs=50, num_samples=10, cpu_per_trial=1, gpu_per_trial=1/10):
    
    # scale max_batch_size by gpu allocation per trial
    max_batch_sizes["image"] = (np.floor(max_batch_sizes["image"]*gpu_per_trial)).astype(int).item()
    max_batch_sizes["pixel"] = (np.floor(max_batch_sizes["pixel"]*gpu_per_trial)).astype(int).item()

    run_config_tune = json_config_to_run_config(tune_config, project_name=project_name)

    # scheduler used to terminate bad trials early, pause trials, and change hyperparameters
    scheduler = ASHAScheduler(
        metric="validation_loss",
        mode="min",
        max_t=20,
        grace_period=1,
        reduction_factor=2,
    )

    # launches hyperparater tuning jobs with given tuning function, scheduler and parameter search space
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tune_preprocessing, fat_depth_training=fat_depth_training, fat_depth_validation=fat_depth_validation, 
                                    apply_transform=apply_transform, device=device, max_batch_sizes=max_batch_sizes, num_epochs=num_epochs),
            resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            reuse_actors=True,
        ),
        param_space=run_config_tune,
        run_config=air.RunConfig(
            verbose=1, 
            name=project_name, 
            local_dir="./raytune",
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                # checkpoint_score_attribute="validation_loss",
                # checkpoint_score_order="min"
                ),
            log_to_file=False,
            # callbacks=[
            #     WandbLoggerCallback(project=project_name)
            # ]
        )
    )
        
    results = tuner.fit()

    return results


def start_hyperparameter_tuning(tune_config, project_name, dataset_training, dataset_validation, device, max_batch_sizes, num_epochs=50, num_samples=10, cpu_per_trial=1, gpu_per_trial=1/10):
    
    # scale max_batch_size by gpu allocation per trial
    max_batch_sizes["image"] = (np.floor(max_batch_sizes["image"]*gpu_per_trial)).astype(int).item()
    max_batch_sizes["pixel"] = (np.floor(max_batch_sizes["pixel"]*gpu_per_trial)).astype(int).item()
    
    run_config_tune = json_config_to_run_config(tune_config, project_name=project_name)
    
    # scheduler used to terminate bad trials early, pause trials, and change hyperparameters
    scheduler = ASHAScheduler(
        metric="validation_loss",
        mode="min",
        max_t=20,
        grace_period=1,
        reduction_factor=2,
    )

    # launches hyperparater tuning jobs with given tuning function, scheduler and parameter search space
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tune_hyperparameters, dataset_training=dataset_training, dataset_validation=dataset_validation, device=device, max_batch_sizes=max_batch_sizes, num_epochs=num_epochs),
            resources={"cpu": cpu_per_trial, "gpu": gpu_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
            reuse_actors=True,
        ),
        param_space=run_config_tune,
        run_config=air.RunConfig(
            verbose=1, 
            name=project_name, 
            local_dir="./raytune",
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                # checkpoint_score_attribute="validation_loss",
                # checkpoint_score_order="min"
                ),
            log_to_file=False
            # callbacks=[
            #     WandbLoggerCallback(project=project_name)
            # ]
        )
    )
        
    results = tuner.fit()

    return results


def tune_preprocessing(config, fat_depth_training, fat_depth_validation, apply_transform, device, max_batch_sizes, num_epochs=50):
    trial_name = [config["scaler_method"], config["reduction_method"], str(config["reduction_components"])]
    trial_name = "_".join(trial_name)

    wandb = setup_wandb(config, trial_name=trial_name, group="preprocessing")
    
    load_as_image = model_accepts_image_data(config)

    if load_as_image:
        max_batch_size = max_batch_sizes["image"]
    else:
        max_batch_size = max_batch_sizes["pixel"]

    dataset_training, dataset_validation = preprocessing_from_config(config, fat_depth_training, fat_depth_validation, apply_transform, load_as_image)
    run_trainer_and_log(config, dataset_training, dataset_validation, device, max_batch_size, wandb=wandb, num_epochs=num_epochs)

def tune_hyperparameters(config, dataset_training, dataset_validation, device, max_batch_sizes, num_epochs=50):
    # trial_name = [config["scaler_method"], config["reduction_method"], str(config["reduction_components"])]
    # trial_name = "_".join(trial_name)

    # wandb = setup_wandb(config, trial_name=trial_name, group="hyperparameter")
    wandb = setup_wandb(config, group="hyperparameter")
    
    load_as_image = model_accepts_image_data(config)

    if load_as_image:
        max_batch_size = max_batch_sizes["image"]
    else:
        max_batch_size = max_batch_sizes["pixel"]

    run_trainer_and_log(config, dataset_training, dataset_validation, device, max_batch_size, wandb=wandb, num_epochs=num_epochs)
            
def run_trainer_and_log(config, dataset_training, dataset_validation, device, max_batch_size, wandb=None, num_epochs=50):
    # general hyperparameters
    batch_size = int(config["batch_size"])
    learning_rate = config["learning_rate"]
    momentum = config["momentum"]

    # data loaders
    train_data_loader = DataLoader(dataset_training, batch_size=batch_size)  
    validate_data_loader = DataLoader(dataset_validation, batch_size=max_batch_size)

    config["input_channels"] = config["reduction_components"]
    config["output_channels"] = 1

    # load model
    model = load_model_from_config(config).to(device)

    #loss and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):
        train_loss, epoch_time = train_epoch(model, train_data_loader, device, optimizer, criterion)
        validation_loss = valiate_epoch(model, validate_data_loader, device, criterion)

        session.report({
            "validation_loss": validation_loss,
            "training_loss": train_loss,
            "epoch_time": epoch_time}
        )

        if wandb:
            wandb.log({
            "validation_loss": validation_loss,
            "training_loss": train_loss,
            "epoch_time": epoch_time}
        )