import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import trochvision.transforms as transforms
from torchvision.models.squeeznet import SqueezeNet, squeezenet1_1
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser('Options for worker batch train')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

random.seed(args.seed)

# for CIFAR-10
num_classes = 10
input_size = 32*32

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
]))
trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

# testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms.Compose([
#     transforms.Resize(input_size),
#     transforms.CenterCrop(input_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
# ]))
# testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

def intialize_net(weight_dict=None):
    model = squeezenet1_1(pretrained=weight_dict is None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    if weight_dict:
        model.load_state_dict(weight_dict)
    return model

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    param_update_dict={}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    ## figure out how to get param updates
    return param_update_dict

def __main__():
    model = initialize_net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.adam(model.parameters(), lr=args.learning_rate)
    train_model(model, trainset_loader, criterion, optimizer, args.epoch)
