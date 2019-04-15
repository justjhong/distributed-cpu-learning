from mpi4py import MPI
from master_ops.pserver import train_from_pretrained
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    pserver = train_from_pretrained(1)
else:
    # exec(open("./worker_batch_ops/train.py").read(), globals())
    subprocess.call(['python3', './worker_batch_ops/train.py', '--rank', str(rank), '--size', str(size), '--num-iter', '2', '--epochs', '1'])
