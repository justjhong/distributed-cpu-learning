import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.squeezenet import *
from hessianflow_nd.optimizer.optm_utils import exp_lr_scheduler
from hessianflow_nd.eigen import get_eigen
import argparse
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description='Various hyperparam settings')
parser.add_argument('--comm-interval', type = int, default = 1, metavar = 'CI',
                    help = 'minibatches until the models synchronize')
parser.add_argument('--batch-mult', type = int, default = 1, metavar = 'BM',
                    help = 'ratio at which batch size will increase relative to eigenvalue decrease')
parser.add_argument('--eig-comm', type=int, default=1, metavar='EC', help = 'Whether communication is used for eigenvalue computation every time')
parser.add_argument('--num-cores', type=int, default=1, metavar='T', help = 'num cores used, but does not do anything, just for file naming')
parser.add_argument('--init-batch-size', type=int, default=64, metavar='BS', help = 'initial batch size')
args = parser.parse_args()

# HessianFlow initialization
large_grad = []
large_ratio = 1
max_large_ratio = 16
max_eig = None
decay_ratio = 2
init_batch_size = args.init_batch_size
batch_update_flag = True
if max_large_ratio == 1:
    batch_update_flag = False

# Define dataset...
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

# Partition dataset among workers using DistributedSampler
train_dataset = datasets.CIFAR10(root='./datasets', train = True, download = True, transform = transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=init_batch_size,drop_last=True, shuffle=True)

test_dataset = datasets.CIFAR10(root='./datasets', train = False, download = True, transform = transform_train)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, drop_last=True, shuffle=True)

hessian_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)

# Build model...
model = squeezenet1_1()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

optimizer.zero_grad()

# Keep track of losses
train_file = "train_loss_comm-{}_bmult-{}_cores-{}_eig-comm-{}_init-batch-size-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm), str(args.init_batch_size))
test_file = "test_loss_comm-{}_bmult-{}_cores-{}_eig-comm-{}_init-batch-size-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm), str(args.init_batch_size))
test_acc_file = "test_acc_comm-{}_bmult-{}_cores-{}_eig-comm-{}_init-batch-size-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm), str(args.init_batch_size))
eig_file = "eig_comm-{}_bmult-{}_cores-{}_eig-comm-{}_init-batch-size-{}".format(str(args.comm_interval), str(args.batch_mult), str(args.num_cores), str(args.eig_comm), str(args.init_batch_size))
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
            if batch_idx * large_ratio % 25 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, batch_idx * len(data), args.init_batch_size * len(train_loader), large_batch_loss))
                train_losses.append((time.clock() - start_time, epoch, batch_idx, loss.item()))
            large_batch_loss = 0
        if batch_idx % 100 == 0:
            model.eval()
            try:
                inputs, labels = next(testset_iterator)
            except StopIteration:
                testset_iterator = iter(test_loader)
                inputs, labels = next(testset_iterator)
            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            accuracy = outputs.data.max(1)[1].eq(labels).sum().item() / outputs.data.shape[0]

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
        eig, _ = get_eigen(model, inputs, labels, criterion, maxIter=10, tol= 1e-2)
        ref_eigs.append(eig)
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
    data_file = "./results/nd_hessian/" + file_name

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


# Make train plot
plot_loss(train_losses, train_file)
# Make test plot
plot_loss(test_losses, test_file)
plot_loss(test_accs, test_acc_file, "Accuracy")
# plot_eigs(ref_eigs, exp_eigs, eig_file)
