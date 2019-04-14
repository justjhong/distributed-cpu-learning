import numpy as np
import torch
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_1
from mpi4py import MPI

class ParameterServer:
    def __init__(state_dict, num_workers, mpi_comm):
        self.state_dict = state_dict
        self.num_workers = num_workers
        self.mpi_comm = mpi_comm

    # Update weights to be average of weights
    def perform_weight_update(self, weights):
        for name in self.state_dict:
            value = 0
            for update in weight_updates:
                if name in update:
                    value += update[name]
            self.state_dict[x].copy_(value / num_workers)

    def request_updates(self):
        ## mpi requests with current self.state_dict

        # requests full weights from all the workers
        # by averaging all the weights, it is the same as averaging gradient update
        updates = self.mpi_comm.gather(None, root = 0)

        self.perform_weight_update(updates)

        # sends weights back to worker nodes
        self.mpi_comm.bcast(self.state_dict, root = 0)

# start training from pretrained parameters
def train_from_pretrained(num_iter):
    model = squeezenet1_1(pretrained=True)

    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size() - 1

    pserver = ParameterServer(dict(list(model.named_parameters)), num_workers, comm)
    for i in range(num_iter):
        pserver.request_updates()
