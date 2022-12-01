import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

import os
import argparse

from model import *
from model.Cnn import Cnn
from utils import progress_bar

import wandb
writer = SummaryWriter() # for tensorboard

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate') # backbone - set small as much when using backbone
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

# DATA
def get_mean(dataset):
  meanRGB = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
  meanR = np.mean([m[0] for m in meanRGB])
  meanG = np.mean([m[1] for m in meanRGB])
  meanB = np.mean([m[2] for m in meanRGB])
  return [meanR, meanG, meanB]

def get_std(dataset):
  stdRGB = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]
  stdR = np.mean([s[0] for s in stdRGB])
  stdG = np.mean([s[1] for s in stdRGB])
  stdB = np.mean([s[2] for s in stdRGB])
  return [stdR, stdG, stdB]

train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

# data augmentation 진행 X -> data augmentation 진행하면 mean, std가 달라짐
train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(get_mean(train_dataset), get_std(train_dataset))])
test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(get_mean(test_dataset), get_std(test_dataset))])
                                      
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# trainsform 
train_dataset.transform = train_transforms
test_dataset.transform = test_transforms

# dataloader 
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=16)

# Model
print('==> Building model..')
# net = ResNet18()
net = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2) # imagenet backbone
# net = Cnn()

num_ftrs = net.fc.in_features   # transfer learning
net.fc = nn.Linear(num_ftrs, 10).cuda() # transfer learning
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print("training")
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({"training loss": train_loss})
    wandb.log({"training accuracy": correct/total})
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', correct/total, epoch)
    

def test(epoch):
    print("validation")
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({"test loss": test_loss})
    wandb.log({"test accuracy": correct/total})
    wandb.watch(net)
    
    # tensorboard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', correct/total, epoch)
    
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def main():
    torch.manual_seed(42)
    torch.multiprocessing.freeze_support()
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
        writer.flush()

def wandb_integration():
    wandb.init(project="cifar10", entity="24bean")
    wandb.config = {
    "learning_rate": 0.01,
    "epochs": 200,
    "batch_size": 128
    }

if __name__=="__main__":
    wandb_integration()
    main()