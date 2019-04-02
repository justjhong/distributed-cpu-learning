import numpy as np
import torch
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_1

num_workers = 5

class ParameterServer:
    def __init__(state_dict, num_workers):
        self.state_dict = state_dict
        self.num_workers = num_workers

    def perform_weight_update(self, weight_updates):
        for update in weight_updates:
            for x in self.state_dict:
                if x in update:
                    self.state_dict[x].add_(update[x])

    def request_updates(self):
        ## mpi requests with current self.state_dict
        updates = None
        self.perform_weight_update(updates)

# start training from pretrained parameters
def train_from_pretrained(num_iter):
    model = squeezenet1_1(pretrained=True)
    pserver = ParameterServer(dict(list(model.named_parameters)), num_workers)
    for i in range(num_iter):
        pserver.request_updates()
