import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim
import torch.utils.data as data
import torchvision.datasets as datasets
import numpy as np
from torchsummary import summary

from mixnet import *


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data', default='data/cifar10', required=False,
                    help='path to ImageNet folder')
parser.add_argument('--dataset', default='cifar10', required=False,
                    help='which dataset to train on')
parser.add_argument('--ngpus', default=0, type=int,
                    help='number of GPUs to use')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--resume', default='', type=str,
                    help='Checkpoint state_dict file to resume training from')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch, args, max_iter):
    """Trains model on ImageNet"""

    # set model to training mode
    model.train()

    batch_iterator = iter(train_loader)
    for iteration in range(0, max_iter):

        images, targets = next(batch_iterator)
        if args.cuda:
            images = Variable(images.cuda())
            # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            targets = Variable(targets.cuda())

        else:
            images = Variable(images)
            # targets = [Variable(ann, volatile=True) for ann in targets]
            targets = Variable(targets)

        tic = time.time()

        # forward prop
        out = model(images)
        print("forward prop done")

        # backprop
        optimizer.zero_grad()

        # calculate loss
        loss = criterion(out, targets)
        print(loss)

        # update weights
        loss.backward()
        optimizer.step()

        toc = time.time()

        if iteration % 1 == 0:
            print('timer: %.4f sec.' % (toc - tic))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')

        if epoch%5 == 0:
            print('here')
            print('Saving state, iter:', iteration)
            torch.save(model.state_dict(), 'weights/' +
                       repr(iteration) + '.pth')

        # torch.save(model.state_dict(),
        #         args.save_folder + '' + args.dataset + '.pth')

def validate(val_loader, model, criterion, args, transform):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():

    model = mixnetV1()
    print(summary(model, (3, 224, 224)))


    if args.cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_weights(args.resume)

    if args.cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    loss_function = nn.CrossEntropyLoss()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,])

    val_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,])

    train_dataset = datasets.CIFAR10(args.data, train=True, transform=train_transform, target_transform=None, download=True)
    val_dataset = datasets.CIFAR10(args.data, train=True, transform=val_transform, target_transform=None, download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler)

    max_iter = len(train_dataset) // args.batch_size
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, loss_function, optimizer, epoch, args, max_iter)
        acc1 = validate(val_loader, model, loss_function, args)
        print(acc1)

if __name__ == '__main__':
    main()
