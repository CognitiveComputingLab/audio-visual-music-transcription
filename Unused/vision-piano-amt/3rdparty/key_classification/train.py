'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.resnet_112_32 import ResNet18
from models.simple import SimpleNet
from models.conv3net import Conv3Net
from torchvision.datasets import ImageFolder 
from dataset import KeyPressDataset 

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from IPython import embed  
from config import  cfg 
from datetime import datetime 
import shutil 

parser = argparse.ArgumentParser(description='PyTorch PianoKeyPress Training')
# Datasets
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=1024, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=512, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[10,25],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--manualSeed', default=100,type=int, help='manual seed')

parser.add_argument('--type',type=str,default='white',choices=['white','black'])

parser.add_argument('--network',type=str,default='conv3net',choices=['simple','resnet','conv3net'])
parser.add_argument('--input_channel',type=int,default=1,choices=[1,3])
parser.add_argument('--data',type=str,default='owndata',choices=['owndata','paperdata'])

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if args.type == 'white':
    if args.data=='paperdata':
        TRAIN_FILE = cfg.Paper_WHITE_TRAIN_FILE 
        TEST_FILE = cfg.Paper_WHITE_VAL_FILE 
    else:
        TRAIN_FILE = cfg.Own_WHITE_TRAIN_FILE 
        TEST_FILE = cfg.Own_WHITE_VAL_FILE 
else:
    if args.data=='paperdata':
        TRAIN_FILE = cfg.Paper_BLACK_TRAIN_FILE 
        TEST_FILE = cfg.Paper_BLACK_VAL_FILE 
    else:
        TRAIN_FILE = cfg.Own_BLACK_TRAIN_FILE 
        TEST_FILE = cfg.Own_BLACK_VAL_FILE 
# Use CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    #start_time = datetime.now().strftime('%m-%d_%H-%M')
    start_time = datetime.now().strftime('%m-%d_%H')
    

    # Data
    print('==> Preparing dataset')
    input_h,input_w = cfg.INPUT_SIZE[args.type][args.data] 
    if args.input_channel==1:
        transform_train=transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((input_h,input_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((input_h,input_w)),
            transforms.ToTensor(),
        ])
    else:
        transform_train=transforms.Compose([
            transforms.Resize((input_h,input_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((input_h,input_w)),
            transforms.ToTensor(),
        ])
    num_classes = 2

    alpha = cfg.ALPHA[args.type][args.data]
    args.checkpoint = os.path.join(cfg.SAVE_MODEL_DIR,
                                    args.data,args.type,
                                    args.network,'h{}_w{}'.format(input_h,input_w),
                                    'alpha_{}'.format(alpha),start_time)
    if os.path.exists(args.checkpoint):
        shutil.rmtree(args.checkpoint)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    train_dataset = KeyPressDataset(TRAIN_FILE,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.train_batch,
                                        shuffle=True,
                                        num_workers=args.workers
                                        )
    test_dataset = KeyPressDataset(TEST_FILE,transform=transform_val)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=args.test_batch,
                                        shuffle=False,
                                        num_workers=args.workers)
    # Model
    print("==> creating model")
    if args.network =='simple':
        model = SimpleNet(input_channel = args.input_channel,type=args.type).to(device)
    elif args.network=='resnet':
        model = ResNet18(input_channel=args.input_channel).to(device)
    else:
        model = Conv3Net(input_channel=args.input_channel,type=args.type).to(device)

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    weights = torch.Tensor([1.0,alpha]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    # Resume
    title = 'piano-key-resnet18'
    print(args)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch)
        test_loss, test_acc = test(testloader, model, criterion, epoch)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        if is_best:
            torch.save(model.state_dict(),os.path.join(args.checkpoint,'best.pth'))
        if epoch>8:
            torch.save(model.state_dict(),os.path.join(args.checkpoint,'epoch_{}_{}.pth'.format(epoch,test_acc)))
    logger.close()
    #logger.plot()
    #savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device).squeeze()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device).squeeze()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
