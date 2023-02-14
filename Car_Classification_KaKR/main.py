import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as model
import torchvision.transforms as transforms
import mydata

import os
import argparse

# Arg
parser = argparse.ArgumentParser(description='Car Classification Training(Pytorch)')
parser.add_argument('--epoch', default=30, type=int, help='Number of Epoch')
parser.add_argument('--lr', default=0.01, type=float, help='Learning Rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from Checkpoint')

args = parser.parse_args()

# Train
def train(dataloader, model, optimizer, loss_func, epoch, device):
    print('============================Train============================')
    model.train()
    train_loss = 0
    total = 0
    correct = 0

    for i, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        if i % 50 == 0:
            print('Iter : [%d/%d] | Train_loss : %0.3f | acc : %0.3f'
                    % (i, len(dataloader), train_loss/total, 100*correct/total))

    print('Iter : [%d/%d] | Train_loss : %0.3f | acc : %0.3f'
            %(i+1, len(dataloader), train_loss/total, 100*correct/total))

    return train_loss
# Test
def test(dataloader, model, loss_func, device):
    print('============================Test============================')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_func(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        if i % 50 == 0:
            print('Iter : [%d/%d] | Test_loss : %0.3f | acc : %0.3f'
                    % (i, len(dataloader), test_loss/total, 100*correct/total))

    acc = 100*correct/total
    print('Iter : [%d/%d] | Test_loss : %0.3f | acc : %0.3f'
            %(i+1, len(dataloader), test_loss/total, acc))

    return acc, test_loss

def main():

    # Transform
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Data
    print('===Data Preparing...')
    data_path = '../../data/car_classification/'
    dataset = mydata.mydataset(data_path, transform=train_transform)
    trainset, testset = torch.utils.data.random_split(dataset, [8000, 2016])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    # Cuda
    print('===Cuda Checking...')
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print(device)

    # Model Building
    print('===Model Building...')
    model_vgg = model.vgg16(pretrained=True)
    model_vgg = model_vgg.to(device)

    for p in model_vgg.features.parameters():
        p.requires_grad = False

    model_vgg.classifier[6].out_features = 196
    
    # checkpoint
    if args.resume:
        print('===Resuming from Checkpoint')
        assert os.path.isdir('checkpoint'), 'Checkpoint 폴더 없음!'
        checkpoint = torch.load('checkpoint/ckpt_vgg.pth')
        model_vgg.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = 0
        start_epoch = 0

    # optim, loss_func
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_vgg.parameters(), lr=args.lr, momentum=0.9)

    train_loss_list, test_loss_list = [], []
    for e in range(start_epoch, start_epoch+args.epoch):
        print('Epoch : %d' % e)
        train_loss = train(trainloader, model_vgg, optimizer, loss_func, e, device)
        acc, test_loss = test(testloader, model_vgg, loss_func, device)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if acc > best_acc:
            print('==Saving...')
            state = {
                'model' : model_vgg.state_dict(),
                'epoch' : e,
                'acc' : acc,
                'train_loss' : train_loss_list,
                'test_loss' : test_loss_list
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_vgg.pth')
            best_acc = acc

if __name__ == '__main__':
    main()