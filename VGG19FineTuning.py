from __future__ import print_function, division
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
import sklearn.metrics as metrics

torch.cuda.empty_cache()
plt.ion()   # interactive mode

class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
		
class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []
        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
		
def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(18, Block, img_channels, num_classes)
	
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated
	
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    #c = torch.rand(20000, 20000).cuda()
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
        
    plt.plot(np.arange(0,num_epochs), training_acc_arr, label='Training Accuracy Curve')       
    plt.plot(np.arange(0,num_epochs), validation_acc_arr, label='Validation Accuracy Curve') 
    plt.ylim([0, 1])
    ax_1.legend(loc=0)
    ax_1.set_ylabel('Accuracy')
    ax_1.set_xlabel('Epoch number')
    fig_1.savefig('accuracy_test.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    fig_2 = plt.figure(figsize=(4, 4))
    ax_2 = fig_2.add_subplot(111)
        
    plt.plot(np.arange(0,num_epochs), training_loss_arr, label='Training Loss Curve')       
    plt.plot(np.arange(0,num_epochs), validation_loss_arr, label='Validation Loss Curve') 
    #plt.ylim([0, 1])
    ax_2.legend(loc=0)
    ax_2.set_ylabel('Loss')
    ax_2.set_xlabel('Epoch number')
    fig_2.savefig('loss_test.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
	
def Test_Model(model):
    model.eval()
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
		
if __name__ == '__main__':
    data_transforms = {
        'Train': transforms.Compose([
            transforms.Resize(256),
            torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.RandomRotation(45),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = r"C:\Users\VINUS\Desktop\MLP\code\Data\Data"
    #data_dir = r"D:\AHALYA\DataScience\semester 2\MLP\Data1010\Data"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['Train', 'Validation', 'Test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                 shuffle=True, num_workers=4)
                  for x in ['Train', 'Validation', 'Test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation']}
    class_names = image_datasets['Train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['Train']))
    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
    #
    # imshow(out, title=[class_names[x] for x in classes])
    #model_ft = models.resnet18(pretrained=True)
    model_ft = models.vgg19(pretrained=True)
     
    for name, child in model_ft.named_children():
        if name in ['avgpool', 'classifier']:
            print(name + ' has been unfrozen.')
            for param in child.parameters():
                param.requires_grad = True

        else:
            for param in child.parameters():
                param.requires_grad = False

    for name, child in model_ft.features.named_children():
        if name in ['10','11','12','13','14','15','16','17','18','19', '20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36']:
            print('features-' + name + ' has been unfrozen.')
            for param in child.parameters():
                param.requires_grad = True
                

    num_ftrs = model_ft.classifier[6].in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    #model_ft.fc = nn.Linear(num_ftrs, 3)
#    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,3))

    features = list(model_ft.classifier.children())[:-1] # Remove last layer
    features.extend([torch.nn.Linear(num_ftrs, 3)])
    model_ft.classifier = torch.nn.Sequential(*features)

    print("------------------------")
    for name, child in model_ft.named_children():
        print(child)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft  = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
    Test_Model(model_ft)