import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.squeezenet import *
import horovod.torch as hvd
from hessianflow.optimizer.optm_utils import exp_lr_scheduler
from hessianflow.eigen import get_eigen
from hessianflow.utils import allreduce_parameters
import matplotlib.pyplot as plt
import time

# Initialize Horovod
hvd.init()
print("Size:{}".format(hvd.size()))
print("Rank:{}".format(hvd.rank()))
print("Local rank:{}".format(hvd.local_rank()))

# HessianFlow initialization
large_grad = []
inner_loop = 0
large_ratio = 1
max_large_ratio = 16
max_eig = None
decay_ratio = 2
init_batch_size = 50
batch_update_flag = True
if max_large_ratio == 1:
    batch_update_flag = False
num_updates = 0

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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, sampler=test_sampler, drop_last=True)

hessian_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

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
train_file = "train_loss_" + str(hvd.rank())
test_file = "test_loss"
test_acc_file = "test_acc"
start_time = time.clock()
train_losses = []
test_losses = []
test_accs = []

criterion = nn.CrossEntropyLoss()
testset_iterator = iter(test_loader)
hessian_iterator = iter(hessian_loader)
for epoch in range(30):
    # optimizer.set_backward_passes_per_step(large_ratio)
    large_batch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx == 2:
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
            allreduce_parameters(model.state_dict())
            optimizer.zero_grad()
            if batch_idx * large_ratio % 25 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(epoch, batch_idx * len(data), len(train_sampler), large_batch_loss))
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
        eig, _ = get_eigen(model, inputs, labels, criterion, maxIter=10, tol= 1e-2)
        if max_eig == None:
            max_eig = eig
        elif eig <= max_eig/decay_ratio:
            max_eig = eig
            prev_ratio = large_ratio
            large_ratio = int(large_ratio*decay_ratio)
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
  plot_file = data_file + "_graph.png"
  f = open(data_file, "w")
  f.write("time, epoch, batch_idx, loss\n")
  for loss in losses:
    f.write("{}, {}, {}, {}\n".format(loss[0], loss[1], loss[2], loss[3]))
  f.close()

  # Plot loss vs time
  plt.plot([loss[0] for loss in losses], [loss[3] for loss in losses], label=file_name)
  plt.ylabel(y_axis)
  plt.xlabel("Time in seconds")
  plt.savefig(plot_file)
  plt.clf()

# Make train plot
plot_loss(train_losses, train_file)

if hvd.rank() == 0:
  # Make test plot
  plot_loss(test_losses, test_file)
  plot_loss(test_accs, test_acc_file, "Accuracy")
