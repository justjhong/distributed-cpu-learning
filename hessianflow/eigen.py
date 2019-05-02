import torch
import math
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import horovod.torch as hvd
from torchvision.models.squeezenet import *

from .utils import *


def get_eigen(model, inputs, targets, criterion, maxIter = 50, tol = 1e-3, comm=True):
    """
    compute the top eigenvalues of model parameters and
    the corresponding eigenvectors.

    change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    If you call this functino during training, remember to change the mode back to training mode.
    model.eval()
    """

    model.eval()
    # torch.no_grad()

    #model_copy = squeezenet1_1(pretrained=False)
    #model_copy.load_state_dict(model.state_dict())
    #optimizer = optim.SGD(model_copy.parameters(), lr=0.001 * hvd.size(), momentum=0.9)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward(create_graph=True)

    params, gradsH = get_params_grad(model)
    v = [torch.randn(p.size()) for p in params]
    v = normalization(v)
    if comm:
        hvd.broadcast_parameters(v, root_rank=0)

    eigenvalue = None

    for i in range(maxIter):
        print(i)
        model.zero_grad()
        Hv = hessian_vector_product(gradsH, params, v)
        if comm:
            handles = []
            for i in range(len(Hv)):
                handles.append(hvd.allreduce_async_(Hv[i], name='reduce random vector update {}'.format(i)))
            for handle in handles:
                hvd.synchronize(handle)
        eigenvalue_tmp = group_product(Hv, v).item()
        v = normalization(Hv)
        if eigenvalue == None:
            eigenvalue = eigenvalue_tmp
        else:
            if abs(eigenvalue-eigenvalue_tmp) < tol:
                if comm:
                    return eigenvalue_tmp, v
            else:
                eigenvalue = eigenvalue_tmp
    if not comm:
        print("{} is here".format(hvd.rank()))
        eigenvalue = torch.FloatTensor([eigenvalue])
        hvd.allreduce_(eigenvalue, name='eigenvalue')
        print("allreduced eigs for rank {}".format(hvd.rank()))
        eigenvalue = float(eigenvalue)
        if hvd.rank() == 0:
            print("No Communication eigenvalue approximated at {}".format(eigenvalue))
    return eigenvalue, v

def get_eigen_full_dataset(model, dataloader, criterion, maxIter = 50, tol = 1e-3):
    """
    compute the top eigenvalues of model parameters and
    the corresponding eigenvectors with a full dataset.
    Notice, this is very expensive.

    change the model to evaluation mode, otherwise the batch Normalization Layer will change.
    If you call this functino during training, remember to change the mode back to training mode.
    """
    model.eval()


    params,_ = get_params_grad(model)
    v = [torch.randn(p.size()) for p in params]
    v = normalization(v)

    batch_size = None
    eigenvalue = None

    for i in range(maxIter):
        THv = [torch.zeros(p.size()) for p in params]
        counter = 0
        for inputs, targets in dataloader:

            if batch_size == None:
                batch_size = targets.size(0)

            if targets.size(0) < batch_size:
                continue

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(create_graph=True)

            params, gradsH = get_params_grad(model)
            Hv = torch.autograd.grad(gradsH, params, grad_outputs = v, only_inputs = True, retain_graph = False)

            THv = [THv1 + Hv1 + 0. for THv1, Hv1 in zip(THv, Hv)]
            counter += 1

        eigenvalue_tmp =group_product(THv,v).item() / float(counter)
        v = normalization(THv)

        if eigenvalue == None:
            eigenvalue = eigenvalue_tmp
        else:
            if abs(eigenvalue-eigenvalue_tmp) < tol:
                return eigenvalue_tmp, v
            else:
                eigenvalue = eigenvalue_tmp

    return eigenvalue, v
