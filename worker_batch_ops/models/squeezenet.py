from __future__ import absolute_import
import torch.nn as nn
from torchvision.models.squeezenet import squeezenet1_1

__all__ = ['squeezenet']

def squeezenet(num_classes=10, pretrained=False):
    model = squeezenet1_1(pretrained=pretrained)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes

