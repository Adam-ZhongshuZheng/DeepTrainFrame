from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.utils.data import SubsetRandomSampler
from torchvision import datasets

from utils import progress_bar
from data_loader import *
from model import *

import pdb

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--max_epoch', default=100, type=int, help='max epoch')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mode', '-m', default='A', type=str, help='train/test mode')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

flod = args.flod
mode = args.mode



# Data
def data_prepare():
    """
    make the dataloader and return them: train, test, validation

    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.7726, 0.6524, 0.8035], [0.0795, 0.1099, 0.0811]),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.7726, 0.6524, 0.8035], [0.0795, 0.1099, 0.0811]),
    ])

    # readpath = os.path.join('data_flods', 'flod' + str(flod))

    dataset = MNISTDataset("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", transform=transform_train)
    dataset_test = MNISTDataset("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", transform=transform_test)

    # To get shuffled dataset
    valid_rate = [0.6, 0.8, 1.0]    # the valid and test part

    indices = list(range(len(dataset)))
    split1 = int(np.floor(valid_rate[0] * len(dataset)))
    split2 = int(np.floor(valid_rate[1] * len(dataset)))
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:split1], indices[split1: split2], indices[split2:]
    train_sampler = SubsetRandomSampler(train_idx)
    vali_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False)
    valiloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=vali_sampler, shuffle=False)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, shuffle=False)

    return trainloader, valiloader, testloader


# Model
def model_prepare():
    """get the model, operator and the loss function.
    """
    print('==> Building model..')
    global best_acc
    global start_epoch

    net = image_model_fc(num_classes=10)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # TO Check the check point.
    if args.resume:
        print('==> Resuming model from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_flod' + str(flod) + '.htnet')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # To adjust the learning rate
    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    # optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, cooldown=10)

    # print('==> parameters to train:')
    # for name, param in net.named_parameters():
    #     print("\t", name)

    return net, optimizer, criterion


def train(epoch, dataloader, net, optimizer, criterion):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, databook in enumerate(dataloader):

        optimizer.zero_grad()

        # to run in the network
        inputs = databook['img']
        targets = databook['lbl']
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.uint8)
        outputs = net(inputs)

        # compute the loss and optimize
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        # statics
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.byte().eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # print('Loss: %.3f | Acc: %.3f (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch, dataloader, net, optimizer, criterion, vali=True):
    """Validation and the test."""
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, databook in enumerate(dataloader):

            # run in the net
            inputs = databook['img']
            targets = databook['lbl']
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.uint8)
            outputs = net(inputs)

            # loss and statics
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.byte().eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save the best model till now. Works in validation
    if vali is True:
        acc = 100. * correct / total
        if acc >= best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_flod' + str(flod) + '.htnet')
            best_acc = acc


if __name__ == '__main__':

    trainloader, valiloader, testloader = data_prepare()
    net, optimizer, criterion = model_prepare()
    for epoch in range(start_epoch, start_epoch+args.max_epoch):

        train(epoch, trainloader, net, optimizer, criterion)
        test(epoch, valiloader, net, optimizer, criterion, vali=True)
#     print(start_epoch)
#     test(start_epoch, testloader, net, optimizer, criterion, mode, vali=False)
