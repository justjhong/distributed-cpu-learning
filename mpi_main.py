from mpi4py import MPI
from master_ops.pserver import train_from_pretrained
from worker_batch_ops.train import train

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("starting param server")
    pserver = train_from_pretrained(comm)
else:
    print("starting worker")
    train(comm, rank, size)
