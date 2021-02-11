import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import make_grid
import os
import random
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, filename='SaveFile.tar'):
    torch.save(state, filename)


# loss function
# if GPU is available set loss function to use GPU

def GetModel():
    # instantiate transfer learning model
    vgg_model = models.vgg19(pretrained=False)

    # set all paramters as trainable
    for param in vgg_model.parameters():
        param.requires_grad = True

    # get input of fc layer
    n_inputs = vgg_model.classifier[6].in_features

    # redefine fc layer / top layer/ head for our classification problem
    vgg_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.4),
                                 nn.Linear(2048, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.4),
                                 nn.Linear(2048, 4),
                                 nn.LogSigmoid())

    # set all paramters of the model as trainable
    for name, child in vgg_model.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = True

    # set model to run on GPU or CPU absed on availibility
    vgg_model.to(device)

    # print the trasnfer learning NN model's architecture
    print(vgg_model)
    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(vgg_model.parameters(), momentum=0.9, lr=3e-4)

    # number of training iterations
    epochs = 30
    return vgg_model

GetModel()

# empty lists to store losses and accuracies
train_losses = []
test_losses = []
train_correct = []
test_correct = []


def TrainModel(train_gen, valid_gen, epochs, model, criterion, optimizer):
    # set training start time
    start_time = time.time()

    # set best_prec loss value as 2 for checkpoint threshold
    best_prec1 = 2

    # empty batch variables
    b = None
    train_b = None
    test_b = None

    # start training
    for i in range(epochs):
        # empty training correct and test correct counter as 0 during every iteration
        trn_corr = 0
        tst_corr = 0

        # set epoch's starting time
        e_start = time.time()

        # train in batches
        for b, (y, X) in enumerate(train_gen):
            # set label as cuda if device is cuda
            X, y = X.to(device), y.to(device)

            # forward pass image sample
            y_pred = model(X.view(-1, 3, 512, 512))
            # calculate loss
            loss = criterion(y_pred.float(), torch.argmax(y.view(32, 4), dim=1).long())

            # get argmax of predicted tensor, which is our label
            predicted = torch.argmax(y_pred, dim=1).data
            # if predicted label is correct as true label, calculate the sum for samples
            batch_corr = (predicted == torch.argmax(y.view(32, 4), dim=1)).sum()
            # increment train correct with correcly predicted labels per batch
            trn_corr += batch_corr

            # set optimizer gradients to zero
            optimizer.zero_grad()
            # back propagate with loss
            loss.backward()
            # perform optimizer step
            optimizer.step()

        # set epoch's end time
        e_end = time.time()
        # print training metrics
        print(
            f'Epoch {(i + 1)} Batch {(b + 1) * 4}\nAccuracy: {trn_corr.item() * 100 / (4 * 8 * b):2.2f} %  Loss: {loss.item():2.4f}  Duration: {((e_end - e_start) / 60):.2f} minutes')  # 4 images per batch * 8 augmentations per image * batch length

        # some metrics storage for visualization
        train_b = b
        train_losses.append(loss)
        train_correct.append(trn_corr)

        X, y = None, None

        # validate using validation generator
        # do not perform any gradient updates while validation
        with torch.no_grad():
            for b, (y, X) in enumerate(valid_gen):
                # set label as cuda if device is cuda
                X, y = X.to(device), y.to(device)

                # forward pass image
                y_val = model(X.view(-1, 3, 512, 512))

                # get argmax of predicted tensor, which is our label
                predicted = torch.argmax(y_val, dim=1).data

                # increment test correct with correcly predicted labels per batch
                tst_corr += (predicted == torch.argmax(y.view(32, 4), dim=1)).sum()

        # get loss of validation set
        loss = criterion(y_val.float(), torch.argmax(y.view(32, 4), dim=1).long())
        # print validation metrics
        print(f'Validation Accuracy {tst_corr.item() * 100 / (4 * 8 * b):2.2f} Validation Loss: {loss.item():2.4f}\n')

        # if current validation loss is less than previous iteration's validatin loss create and save a checkpoint
        is_best = loss < best_prec1
        best_prec1 = min(loss, best_prec1)
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

        # some metrics storage for visualization
        test_b = b
        test_losses.append(loss)
        test_correct.append(tst_corr)

    # set total training's end time
    end_time = time.time() - start_time

    # print training summary
    print("\nTraining Duration {:.2f} minutes".format(end_time / 60))
    print("GPU memory used : {} kb".format(torch.cuda.memory_allocated()))
    print("GPU memory cached : {} kb".format(torch.cuda.memory_cached()))
