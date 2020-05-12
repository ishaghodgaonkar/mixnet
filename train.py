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
import matplotlib.pyplot as plt
from sklearn import metrics
from mixnet import *


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data', default='/local/a/cam2/data/ILSVRC2012_Classification/', required=False,
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
parser.add_argument('--batch_size', default=8, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--resume', default='', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--gpu', default=None, type=str)

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
        print(images.shape, targets)
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
        # print(images.shape)
        # print(targets)
        # print(targets.shape, out.shape)
        # print(targets[0], out[0])
        # backprop
        optimizer.zero_grad()

        # calculate loss
        loss = criterion(out, targets)

        # update weights
        loss.backward()
        optimizer.step()

        toc = time.time()

        record = open('record.txt', 'a')

        if iteration % 1 == 0:
            print('timer: %.4f sec.' % (toc - tic))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
            str_to_write = 'iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data) + '\n'
            record.write(str_to_write)

        record.close()

    if epoch%5 == 0:
        print('here')
        print('Saving state, iter:', epoch)
        torch.save(model.state_dict(), 'weights/' +
                   repr(epoch) + '.pth')
        
def validate(val_loader, model, criterion, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        all_outputs = []
        all_targets = []

        for i, (images, target) in enumerate(val_loader):
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
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

            output_list = []
        #    for each in output:
         #       output_list.append(list(each))
            output_list = [list(x) for x in output]  
            output_list = [x.index(max(x)) for x in output_list]
            target = list(target)
            target = [int(x) for x in target]
            
            all_outputs.extend(output_list)
            all_targets.extend(target)
    
    conf_matrix = metrics.confusion_matrix(all_targets, all_outputs)

    print(conf_matrix)

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
    print('here')
    print(summary(model, (3, 224, 224)))

    # Visualize kernels
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            print(m)
            print(m.weight.data.shape)

    if args.cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_weights(args.resume)

    if args.cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    loss_function = nn.CrossEntropyLoss()

    normalize = transforms.Normalize(mean=[0.456],
                                     std=[0.224])

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                normalize,])

    val_transform = transforms.Compose([
                transforms.CenterCrop(224),
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                normalize,])
    dataset = datasets.ImageFolder('/local/a/cam2/data/ILSVRC2012_Classification/train/', transform=train_transform)

    #dataset = datasets.ImageFolder(args.data, transform = train_transform)
    len_dataset = len(dataset)
    len_train = int(0.8*len(dataset))
    len_val = len_dataset - len_train
    print(len_train)
    print(len_val)
    train_set, val_set = torch.utils.data.random_split(dataset, [len_train, len_val])
    print(train_set)
    print(val_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True,
       )


    max_iter = len(train_set) // args.batch_size
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, loss_function, optimizer, epoch, args, max_iter)
        acc1 = validate(val_loader, model, loss_function, args)
        print(acc1)

if __name__ == '__main__':
    main()
