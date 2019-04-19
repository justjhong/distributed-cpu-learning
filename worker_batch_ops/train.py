#*
# @file ABSA training driver based on arxiv:1810.01021
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of HessianFlow library.
#
# HessianFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HessianFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HessianFlow.  If not, see <http://www.gnu.org/licenses/>.
#*
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.models.squeezenet import *
from mpi4py import MPI
# from torchvision.models.resnet import *

from worker_batch_ops.utils import *
from worker_batch_ops.hessianflow.optimizer.progressbar import progress_bar
from worker_batch_ops.hessianflow.optimizer.optm_utils import exp_lr_scheduler, test

def train(mpi_comm, rank, size, update_interval=1, batch_size=128, test_batch_size=200, epochs=10, lr=0.1, lr_decay=0.2, lr_decay_epoch=[30,60,90], seed=1, arch="SqueezeNet", depth=20):
    # set random seed to reproduce the work
    torch.manual_seed(seed)

    # Rank 0 is parameter server, else is worker
    if rank > 0:
        # get dataset
        train_loader, test_loader = getData(name = 'cifar10', train_bs = batch_size, test_bs = test_batch_size)

        transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

        trainset = datasets.CIFAR10(root='../datasets', train = True, download = True, transform = transform_train)
        hessian_loader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True)


    # get model and optimizer
    model_list = {
        'SqueezeNet': lambda: squeezenet1_1(),
        'ResNet': lambda: resnet(depth = depth),
    }

    model = model_list[arch]()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    ########### training
    if rank > 0:
        # large_ratio = max_large_ratio
        # large_ratio originally used to prevent large batches from overloading memory, instead memory split amongst machines
        for epoch in range(1, epochs + 1):
            print('\nCurrent Epoch: ', epoch)
            print('\nTraining')
            train_loss = 0.
            total_num = 0.
            correct = 0.

            for batch_idx, (data, target) in enumerate(train_loader):
                if target.size(0) < 128:
                    continue
                model.train()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                train_loss += loss.item()*target.size(0)
                total_num += target.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                optimizer.step()
                optimizer.zero_grad()

                if batch_idx % update_interval == update_interval - 1:
                    # update server
                    weights = model.state_dict()
                    mpi_comm.gather(weights, root = 0)
                    mpi_comm.Barrier()
                    # Then receive averaged model from param server, change current model to param server model.
                    new_state_dict = None
                    print("Waiting on broadcast")
                    mpi_comm.Barrier()
                    model.load_state_dict(new_state_dict)

                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / total_num,
                                100. * correct / total_num, correct, total_num))

            if epoch in lr_decay_epoch:
                exp_lr_scheduler(optimizer, decay_ratio=lr_decay_ratio)

            # test on master instead
            # test(model, test_loader)
