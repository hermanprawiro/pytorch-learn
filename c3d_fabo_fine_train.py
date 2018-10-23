import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from models.c3d import C3D
from utils.fabo_dataset import FABO

BATCH_SIZE = 16
checkpoint_path = './checkpoints/c3d_fabo_fine.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ('sadness', 'happiness', 'surprised', 'boredom', 'disgust', 'fear', 'anger', 'uncertainty', 'puzzlement', 'anxiety')
# classes = ('sadness', 'happiness', 'surprised', 'boredom', 'disgust', 'fear', 'uncertainty', 'anxiety')
num_classes = len(classes)

def main():
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.ToTensor(),
        # transforms.ColorJitter()
    ])

    trainset = FABO(train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = FABO(train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    net = C3D(487)
    net.load_state_dict(torch.load('c3d.pickle'))
    for param in net.parameters():
        param.requires_grad = False
    net.fc8 = nn.Linear(4096, num_classes)
    net.to(device)

    w_loss = torch.from_numpy(np.array([1/29, 1/34, 1/30, 1/66, 1/46, 1/35, 1/103, 1/38, 1/114, 1/51], dtype=np.float32)).to(device)
    w_loss = w_loss**2
    # w_loss = torch.from_numpy(np.array([29, 34, 30, 66, 46, 35, 103, 38, 114, 51], dtype=np.float32)).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=w_loss)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = optim.Adam(net.parameters())
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=1e-5)

    resume = True
    # resume = False
    start_epoch = 0

    if resume:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Checkpoint file not found')

    print('Start training on epoch:', start_epoch+1)
    train(trainloader, net, criterion, optimizer, start_epoch=start_epoch, num_epoch=10, report_step=5)
    test(testloader, net, criterion)
    train(trainloader, net, criterion, optimizer, start_epoch=start_epoch, num_epoch=500, report_step=5)
    test(testloader, net, criterion)

def train(train_loader, model, criterion, optimizer, start_epoch=0, num_epoch=2, report_step=100):
    model.train()
    for epoch in range(start_epoch, start_epoch+num_epoch):
        train_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            # print(outputs)
            print(predicted, labels)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if i % report_step == report_step-1: # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f | acc: %.3f% % (%d/%d)' % (epoch + 1, i + 1, train_loss / report_step, 100.*correct/total, correct, total))
                train_loss = 0.0
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint_path)
    print('Finished Training')

def test(test_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # print(outputs)
            _, predicted = outputs.max(1)
            c = (predicted == labels).squeeze()
            for i in range(BATCH_SIZE):
                if i < len(labels):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %.3f% %' % (
        100 * correct / total))
    for i in range(10):
        print('Accuracy of %s : %.3f% %' % (classes[i], 100 * class_correct[i] / class_total[i]))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()