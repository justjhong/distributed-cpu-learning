import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.squeezenet import *
import horovod.torch as hvd

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

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

# Build model...
model = squeezenet1_1()

optimizer = optim.SGD(model.parameters(), lr=0.001 * hvd.size(), momentum=0.9)

# Add Horovod Distributed Optimizer
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
optimizer.zero_grad()

# Broadcast parameters from rank 0 to all other processes.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

criterion = nn.CrossEntropyLoss()
testset_iterator = iter(test_loader)
for epoch in range(1):
   for batch_idx, (data, target) in enumerate(train_loader):
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       if batch_idx % 25 == 0:
           print('Train Epoch: {} [{}/{}]\tLoss: {}'.format(
               epoch, batch_idx * len(data), len(train_sampler), loss.item()))
       if batch_idx % 100 == 0:
           try:
               inputs, labels = next(testset_iterator)
           except StopIteration:
               testset_iterator = iter(testset_loader)
               inputs, labels = next(testset_iterator)
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           accuracy = outputs.data.max(1)[1].eq(labels).sum().item() / outputs.data.shape[0]

           loss = metric_average(loss, 'avg_loss')
           accuracy = metric_average(accuracy, 'avg_accuracy')

           if hvd.rank() == 0:
               print("Test loss is {}".format(loss))
               print("Test accuracy is {}".format(accuracy))
