import torch
import numpy as np

def valiate_epoch(model, data_loader, criterion):
    model.eval()
    validation_loss = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            output = model(inputs)
            loss = criterion(output,labels)
            validation_loss.append(loss.item())

    validation_loss = np.mean(validation_loss)