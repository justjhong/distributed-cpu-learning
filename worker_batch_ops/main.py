import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_1
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
input_size = 32

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
]))
# trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
# trainset_size = len(trainset)

# for subset of trainset
sample_indices = torch.randperm(len(trainset))[:args.batch_size*2]
sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SubsetRandomSampler(sample_indices), args.batch_size, drop_last=True)
trainset_loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler)
trainset_size = len(sample_indices)


# testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
# ]))
# testset_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

def initialize_net(weight_dict=None):
    model = squeezenet1_1(pretrained=weight_dict is None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes
    if weight_dict:
        model.load_state_dict(weight_dict)
    return model

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    cur_param_update_dict=dict(list(model.named_parameters()))
    param_update_dict={x: cur_param_update_dict[x].clone() for x in cur_param_update_dict}
    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for batch_i, (inputs, labels) in enumerate(dataloader):
            print('Batch {}/{}'.format(batch_i + 1, trainset_size // args.batch_size))
            optimizer.zero_grad()
            print(" opt")
            outputs = model(inputs)
            print(" forward done")
            loss = criterion(outputs, labels)
            print(" loss calc")
            loss.backward()
            print(" backward done")
            optimizer.step()
            print(" opt step done")
    trained_dict=dict(list(model.named_parameters()))
    for x in trained_dict:
        if x in param_update_dict:
            param_update_dict[x].sub_(trained_dict[x])
            param_update_dict[x].mul_(-1)
    return param_update_dict

def main():
    model = initialize_net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    param_update_dict = train_model(model, trainset_loader, criterion, optimizer, args.epoch)
    print(param_update_dict)

if __name__ == '__main__':
    main()
