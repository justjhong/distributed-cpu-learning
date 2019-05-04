import torch
import math
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision.models.squeezenet import *

from .utils import *

def get_eigen(model, inputs, targets, criterion, maxIter = 50, tol = 1e-3):
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
    eigenvalue = None

    for i in range(maxIter):
        print(i)
        model.zero_grad()
        Hv = hessian_vector_product(gradsH, params, v)
        eigenvalue_tmp = group_product(Hv, v).item()
        v = normalization(Hv)
        if eigenvalue == None:
            eigenvalue = eigenvalue_tmp
        else:
            if abs(eigenvalue-eigenvalue_tmp) < tol:
                return eigenvalue_tmp, v
            else:
                eigenvalue = eigenvalue_tmp
    return eigenvalue, v

