from __future__ import print_function, division
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.preprocessing import minmax_scale
import sklearn.metrics as metrics
import cv2 as cv2
import random
random.seed(1)


from CreateSeprateTestTrainFolders import CreateTrainValidationSliptFolder, CreateTestFolder

batch_size = 16
n_epocs = 30

SETA = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\SetA"
SETB = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\SetB"
SETC = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData\SetC"

def MinMaxNormalisation(A):
    for i in range(len(A)):
        image = (A[i].cpu().data.numpy()).transpose((1, 2, 0))
        foo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized = (foo-foo.min())/(foo.max()-foo.min())
        stacked_img = np.stack((normalized,) * 3, axis=-1)
        traspose = stacked_img.transpose((2, 0, 1))
        A[i] = torch.from_numpy(traspose)

    return A;

def train_model(model, criterion, optimizer, scheduler, num_epochs, index):
    c = torch.rand(20000, 20000).cuda()
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    training_acc_arr = np.zeros(num_epochs)
    validation_acc_arr = np.zeros(num_epochs)
    training_loss_arr = np.zeros(num_epochs)
    validation_loss_arr = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = MinMaxNormalisation(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'Train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'Train':
                training_acc_arr[epoch] = epoch_acc
                training_loss_arr[epoch] = epoch_loss
            if phase == 'Validation':
                validation_acc_arr[epoch] = epoch_acc
                validation_loss_arr[epoch] = epoch_loss
            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since

    fig_1 = plt.figure(figsize=(4, 4))
    ax_1 = fig_1.add_subplot(111)

    plt.plot(np.arange(0, num_epochs), training_acc_arr, label='Training Accuracy Curve')
    plt.plot(np.arange(0, num_epochs), validation_acc_arr, label='Validation Accuracy Curve')
    plt.ylim([0, 1])
    ax_1.legend(loc=0)
    ax_1.set_ylabel('Accuracy')
    ax_1.set_xlabel('Epoch number')
    fig1Path = 'accuracy_test{}.pdf'.format(index)
    fig_1.savefig(fig1Path, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format='pdf',
                  transparent=False, bbox_inches=None, pad_inches=0.1,
                  frameon=None, metadata=None)

    fig_2 = plt.figure(figsize=(4, 4))
    ax_2 = fig_2.add_subplot(111)

    plt.plot(np.arange(0, num_epochs), training_loss_arr, label='Training Loss Curve')
    plt.plot(np.arange(0, num_epochs), validation_loss_arr, label='Validation Loss Curve')
    # plt.ylim([0, 1])
    ax_2.legend(loc=0)
    ax_2.set_ylabel('Loss')
    ax_2.set_xlabel('Epoch number')
    fig2Path = 'loss_test{}.pdf'.format(index)
    fig_2.savefig(fig2Path, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format='pdf',
                  transparent=False, bbox_inches=None, pad_inches=0.1,
                  frameon=None, metadata=None)

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def Test_Model(model):
    model.eval()
    with torch.no_grad():
        y_test = []
        y_predict = []
        for i, (inputs, labels) in enumerate(dataloaders['Validation']):
            inputs = MinMaxNormalisation(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_test.extend(labels.cpu().data.numpy())
            y_predict.extend(preds.cpu().data.numpy())
        print("#######################################################################")
        print("**************************Validation conf_matrix*************************")
        conf_matrix = metrics.confusion_matrix(np.array(y_test), np.array(y_predict))
        print(conf_matrix)
        acc = metrics.accuracy_score(np.array(y_test), np.array(y_predict))

        print("**************************Validation Accuracy*************************")
        print("Acc: {:.4f}".format(acc))
        print("*************************Validation F1 score***************************")
        f1score =  metrics.f1_score(np.array(y_test), np.array(y_predict), average=None)
        print(f1score)
        print("F1 score:".format(f1score))
        y_test = []
        y_predict = []
        for i, (inputs, labels) in enumerate(dataloaders['Test']):
            inputs = MinMaxNormalisation(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_test.extend(labels.cpu().data.numpy())
            y_predict.extend(preds.cpu().data.numpy())
        print("#######################################################################")
        print("**************************Test conf_matrix*************************")
        conf_matrix = metrics.confusion_matrix(np.array(y_test), np.array(y_predict))
        print(conf_matrix)
        acc = metrics.accuracy_score(np.array(y_test), np.array(y_predict))
        print("**************************Test Accuracy*************************")
        print("Acc: {:.4f}".format(acc))
        print("*************************Test F1 score***************************")
        f1score = metrics.f1_score(np.array(y_test), np.array(y_predict), average=None)
        print(f1score)
        print("F1 score:".format(f1score))


if __name__ == '__main__':
    j = 1
    totalAccuracy = np.zeros(4)
    # for i in [(SETC, SETA, SETB), (SETA, SETC, SETB), (SETB, SETA, SETC)]:
    # accuracy =[]
    # CreateTrainValidationSliptFolder(i[1],i[2], 0)
    # CreateTestFolder(i[0])
    j = j + 1
    data_transforms = {
        'Train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = r"D:\MSC\Sem2\MLP\BrainTumorClassification\brainTumorData"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['Train', 'Validation', 'Test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['Train', 'Validation', 'Test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation']}
    class_names = image_datasets['Train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['Train']))
    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])
    model_ft = models.resnet18(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    print(model_ft)
    # for name, child in model_ft.named_children():
    #     if name in ['fc', 'avgpool', 'layer4', 'layer3', 'layer2', 'layer1', 'maxpool', 'relu', 'bn1', 'conv1']:
    #         print(name + ' has been unfrozen.')
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False
    # for param in model_ft.parameters():
    # print(param.requires_grad)
    # model_ft = ResNet18(img_channels=3, num_classes=3)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 3)
    print("------------------------")

    # for name, child in model_ft.named_children():
    #     print(name)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs= n_epocs, index= j)
    fileToSave = 'my_model_weights{}.h5'.format(j)
    torch.save(model_ft.state_dict(), 'my_model_weights.h5')
    Test_Model(model_ft)



