from __future__ import print_function, division

import gc
import os
import time
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import sklearn.metrics as metrics

data_dir = "alien_pred"
input_shape = 224
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

if __name__ == '__main__':
    data_transforms = {
        'Train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'Validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'Test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    data_dir = r"D:\MSC\Sem2\IVC\ClassificationTask\brainTumorData"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['Train', 'Validation', 'Test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                 shuffle=True, num_workers=4)
                  for x in ['Train', 'Validation', 'Test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation']}
    class_names = image_datasets['Train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    images, labels = next(iter(dataloaders['Train']))

    rows = 4
    columns = 4
    fig = plt.figure()
    for i in range(1):
        fig.add_subplot(rows, columns, i + 1)
        plt.title(class_names[labels[i]])
        img = images[i].numpy().transpose((1, 2, 0))
        img = std * img + mean
        plt.imshow(img)
    plt.show()

    ## Load the model based on VGG19
    vgg_based = torchvision.models.vgg19(pretrained=False)

    ## freeze the layers
    # for param in vgg_based.parameters():
    #     param.requires_grad = False

    # Modify the last layer
    number_features = vgg_based.classifier[6].in_features
    features = list(vgg_based.classifier.children())[:-1]  # Remove last layer
    features.extend([torch.nn.Linear(number_features, len(class_names))])
    vgg_based.classifier = torch.nn.Sequential(*features)

    vgg_based = vgg_based.to(device)

    print(vgg_based)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg_based.parameters(), lr=0.001, momentum=0.9)


    def train_model(model, criterion, optimizer, num_epochs=25):
        c = torch.rand(20000, 20000).cuda()
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        since = time.time()
        best_acc = 0.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            #set model to trainable
            for phase in ['Train', 'Validation']:
                if phase == 'Train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
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

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'Validation' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        return model


    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        # fig = plt.figure()

        with torch.no_grad():
            y_test = []
            y_predict = []
            for i, (inputs, labels) in enumerate(dataloaders['Test']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_test.extend(labels.cpu().data.numpy())
                y_predict.extend(preds.cpu().data.numpy())
            conf_matrix = metrics.confusion_matrix(np.array(y_test), np.array(y_predict))
            print(conf_matrix)
            acc = metrics.accuracy_score(np.array(y_test), np.array(y_predict))
            print("Accuracy = {}".format(acc))


    vgg_based = train_model(vgg_based, criterion, optimizer_ft, num_epochs=25)

    visualize_model(vgg_based)
