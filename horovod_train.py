import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.squeezenet import *
from hessianflow.utils import allreduce_parameters, metric_average
import horovod.torch as hvd
import matplotlib.pyplot as plt
import argparse
import time

parser = argparse.ArgumentParser(description='Various hyperparam settings')
parser.add_argument('--comm-interval', type = int, default = 1, metavar = 'CI',
                    help = 'minibatches until the models synchronize')
parser.add_argument('--num-cores', type=int, default=16, metavar='T', help = 'num cores used, but does not do anything, just for file naming')
args = parser.parse_args()

# Initialize Horovod
hvd.init()
print("Size:{}".format(hvd.size()))
print("Rank:{}".format(hvd.rank()))
print("Local rank:{}".format(hvd.local_rank()))

# Define dataset...
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='./datasets', train = True, download = True, transform = transform_train)

# Partition dataset among workers using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, sampler=train_sampler)

test_dataset = datasets.CIFAR10(root='./datasets', train = False, download = True, transform = transform_train)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, sampler=test_sampler)

# Build model...
model = squeezenet1_1()

optimizer = optim.SGD(model.parameters(), lr=0.001 * hvd.size(), momentum=0.9)
# optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
optimizer.zero_grad()

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Keep track of losses
train_file = "train_loss_cores-{}_comm-{}".format(str(args.num_cores), str(args.comm_interval))
test_file = "test_loss_cores-{}_comm-{}".format(str(args.num_cores), str(args.comm_interval))
test_acc_file = "test_acc_cores-{}_comm-{}".format(str(args.num_cores), str(args.comm_interval))
start_time = time.clock()
train_losses = []
test_losses = []
test_accs = []

criterion = nn.CrossEntropyLoss()
testset_iterator = iter(test_loader)
num_updates = 0
for epoch in range(30):
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        num_updates += 1
        if num_updates % args.comm_interval == args.comm_interval - 1:
            allreduce_parameters(model.state_dict())
        if batch_idx % 25 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
                epoch, batch_idx * len(data), len(train_sampler), loss.item()))
            train_losses.append((time.clock() - start_time, epoch, batch_idx, loss.item()))
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

def plot_loss(losses, file_name, y_axis = "Loss"):
  data_file = "./results/nonhessian/" + file_name
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
