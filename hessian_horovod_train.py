import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.squeezenet import *
import horovod.torch as hvd
from hessianflow.optimizer.optm_utils import exp_lr_scheduler
from hessianflow.eigen import get_eigen
from hessianflow.utils import allreduce_parameters, metric_average
import argparse
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description='Various hyperparam settings')
parser.add_argument('--comm-interval', type = int, default = 1, metavar = 'CI',
                    help = 'minibatches until the models synchronize')
parser.add_argument('--batch-mult', type = int, default = 1, metavar = 'BM',
                    help = 'ratio at which batch size will increase relative to eigenvalue decrease')
parser.add_argument('--eig-comm', type=int, default=1, metavar='EC', help = 'Whether communication is used for eigenvalue computation every time')
parser.add_argument('--num-cores', type=int, default=16, metavar='T', help = 'num cores used, but does not do anything, just for file naming')
args = parser.parse_args()

# Initialize Horovod
hvd.init()
print("Size:{}".format(hvd.size()))
print("Rank:{}".format(hvd.rank()))
print("Local rank:{}".format(hvd.local_rank()))

# HessianFlow initialization
large_grad = []
large_ratio = 1
max_large_ratio = 16
max_eig = None
decay_ratio = 2
init_batch_size = 64
batch_update_flag = True
if max_large_ratio == 1:
    batch_update_flag = False

# Define dataset...
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

# Partition dataset among workers using DistributedSampler
train_dataset = datasets.CIFAR10(root='./datasets', train = True, download = True, transform = transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=init_batch_size, sampler=train_sampler, drop_last=True)

test_dataset = datasets.CIFAR10(root='./datasets', train = False, download = True, transform = transform_train)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, sampler=test_sampler, drop_last=True)

hessian_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)

# Build model...
model = squeezenet1_1()

optimizer = optim.SGD(model.parameters(), lr=0.001 * hvd.size(), momentum=0.9)

# Add Horovod Distributed Optimizer
# optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
optimizer.zero_grad()

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Keep track of losses
train_file = "train_loss_comm-{}_bmult-{}_cores-{}_eig-comm-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm))
test_file = "test_loss_comm-{}_bmult-{}_cores-{}_eig-comm-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm))
test_acc_file = "test_acc_comm-{}_bmult-{}_cores-{}_eig-comm-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm))
eig_file = "eig_comm-{}_bmult-{}_cores-{}_eig-comm-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm))
start_time = time.clock()
train_losses = []
test_losses = []
test_accs = []
ref_eigs = []
exp_eigs = []

criterion = nn.CrossEntropyLoss()
testset_iterator = iter(test_loader)
hessian_iterator = iter(hessian_loader)
inner_loop = 0
num_updates = 0
for epoch in range(30):
    # optimizer.set_backward_passes_per_step(large_ratio)
    large_batch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 0:
        #     break
        inner_loop += 1

        model.train()
        output = model(data)
        loss = criterion(output, target)
        large_batch_loss += loss.item()
        loss.backward()
        if inner_loop % large_ratio == 0:
            num_updates += 1
            optimizer.step()
            optimizer.zero_grad()
            if num_updates % args.comm_interval == args.comm_interval - 1:
                allreduce_parameters(model.state_dict())
            if batch_idx * large_ratio % 25 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, batch_idx * len(data), len(train_sampler), large_batch_loss))
                cur_batch_loss = loss.item()
                hvd.allreduce_(cur_batch_loss)
                train_losses.append((time.clock() - start_time, epoch, batch_idx, cur_batch_loss))
            large_batch_loss = 0
        if batch_idx % 100 == 0:
            model.eval()
            try:
                inputs, labels = next(testset_iterator)
            except StopIteration:
                testset_iterator = iter(test_loader)
                inputs, labels = next(testset_iterator)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accuracy = outputs.data.max(1)[1].eq(labels).sum().item() / outputs.data.shape[0]

            loss = metric_average(loss, 'avg_loss')
            accuracy = metric_average(accuracy, 'avg_accuracy')

            if hvd.rank() == 0:
                print("Test loss is {}".format(loss))
                print("Test accuracy is {}".format(accuracy))
                test_losses.append((time.clock() - start_time, epoch, batch_idx, loss))
                test_accs.append((time.clock() - start_time, epoch, batch_idx, accuracy))

    if batch_update_flag:
        try:
            inputs, labels = next(hessian_iterator)
        except StopIteration:
            hessian_iterator = iter(hessian_loader)
            inputs, labels = next(hessian_iterator)
        eig, _ = get_eigen(model, inputs, labels, criterion, maxIter=10, tol= 1e-2, comm=args.eig_comm)
        ref_eigs.append(eig)
        # for comparison for no communication averaging
        # exp_eig, exp_ = get_eigen(model, inputs, labels, criterion, maxIter=10, tol= 1e-2, comm=False)
        # exp_eigs.append(exp_eig)
        if max_eig == None:
            max_eig = eig
        elif eig <= max_eig/decay_ratio:
            max_eig = eig
            prev_ratio = large_ratio
            large_ratio = int(large_ratio*decay_ratio*args.batch_mult)
            # adv_ratio /= decay_ratio
            if large_ratio  >= max_large_ratio:
                large_ratio = max_large_ratio
                batch_update_flag = False
            optimizer = exp_lr_scheduler(optimizer, decay_ratio = large_ratio/prev_ratio)
        print("Eigenvalue approximated at {}. Updated batch size is {}".format(eig, init_batch_size * large_ratio))
    # if epoch in lr_decay_epoch:
    #     optimizer = exp_lr_scheduler(optimizer, decay_ratio = lr_decay_ratio)

def plot_loss(losses, file_name, y_axis = "Loss"):
    data_file = "./results/hessian/" + file_name

    f = open(data_file, "w")
    f.write("time, epoch, batch_idx, loss\n")
    for loss in losses:
        f.write("{}, {}, {}, {}\n".format(loss[0], loss[1], loss[2], loss[3]))
    f.close()

def plot_eigs(ref_eigs, exp_eigs, file_name):
    data_file = "./results/hessian_eigs/" + file_name
    f = open(data_file, "w")
    f.write("ref_eig, exp_eig\n")
    for ref_eig, exp_eig in zip(ref_eigs, exp_eigs):
        f.write("{}, {}\n".format(ref_eig, exp_eig))
    f.close()


if hvd.rank() == 0:
    # Make train plot
    plot_loss(train_losses, train_file)
    # Make test plot
    plot_loss(test_losses, test_file)
    plot_loss(test_accs, test_acc_file, "Accuracy")
    # plot_eigs(ref_eigs, exp_eigs, eig_file)
