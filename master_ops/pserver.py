import numpy as np
import torch
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_1
from mpi4py import MPI

class ParameterServer:
    def __init__(self, state_dict, num_workers, mpi_comm):
        self.state_dict = state_dict
        self.num_workers = num_workers
        self.mpi_comm = mpi_comm

    # Update weights to be average of weights
    def perform_weight_update(self, weights):
        for name in self.state_dict:
            value = 0
            for update in weights:
                if name in update:
                    value += update[name]
            self.state_dict[x].copy_(value / num_workers)

    def request_updates(self):
        ## mpi requests with current self.state_dict

        # requests full weights from all the workers
        # by averaging all the weights, it is the same as averaging gradient update
        weights = None
        print("Waiting on weights")
        self.mpi_comm.Barrier()
        print("Gathered weights")

        self.perform_weight_update(weights)

        # sends weights back to worker nodes
        new_state_dict = self.state_dict
        self.mpi_comm.bcast(new_state_dict, root = 0)
        self.mpi_comm.Barrier()
        print("Broadcasted weights")

# start training from pretrained parameters
def train_from_pretrained(comm):
    print("hello")
    model = squeezenet1_1(pretrained=True)

    num_workers = comm.Get_size() - 1

    pserver = ParameterServer(model.state_dict(), num_workers, comm)
    for i in range(5):
        pserver.request_updates()
    return pserver
