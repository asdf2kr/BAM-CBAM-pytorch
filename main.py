import os
import time
import shutil
import parser
import argparse

import torch
import torch.nn
import torch.optim
import torchvision.models as models

import Models.resnet as resnet

from utils import prepare_dataloaders
from tqdm import tqdm
'''
    reference:
        pytorch, torchvision
        conda install -c conda-forge torchvision
'''

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='Implement image classification on ImageNet datset using pytorch')
    parser.add_argument('--arch', default='bam', type=str, help='Attention Model (bam, cbam)')
    parser.add_argument('--backbone', default='resnet50', type=str, help='backbone classification model (resnet(18, 34, 50, 101, 152)')
    parser.add_argument('--epoch', default=1, type=int, help='start epoch')
    parser.add_argument('--n_epochs', default=1000, type=int, help='numeber of total epochs to run')
    parser.add_argument('--batch', default=256, type=int, help='mini batch size (default: 256)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--save_directory', default='trained.chkpt', type=str, help='path to latest checkpoint')
    parser.add_argument('--workers', default=8, type=int, help='num_workers')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    parser.add_argument('--datasets', default='CIFAR100', type=str, help='classification dataset  (CIFAR10, CIFAR100, ImageNet)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--save', default='trained', type=str, help='trained.chkpt')
    parser.add_argument('--save_multi', default='trained_multi', type=str, help='trained_multi.chkpt')
    parser.add_argument('--evaluate', default=False, type=bool, help='evaluate')
    parser.add_argument('--reduction_ratio', default=16, type=int, help='reduction_ratio')
    parser.add_argument('--dilation_value', default=4, type=int, help='reduction_ratio')
    args = parser.parse_args()
    args.arch = args.arch.lower()
    args.backbone = args.backbone.lower()
    args.datasets = args.datasets.lower()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    # To-do: Write a code relating to seed.

    # use gpu or multi-gpu or not.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_multi_gpu = torch.cuda.device_count() > 1
    print('[Info] device:{} use_multi_gpu:{}'.format(device, use_multi_gpu))

    if args.datasets == 'cifar10':
        num_classes = 10
    elif args.datasets == 'cifar100':
        num_classes = 100
    elif args.datasets == 'imagenet':
        num_classes = 1000

    # load the data.
    print('[Info] Load the data.')
    train_loader, valid_loader, train_size, valid_size = prepare_dataloaders(args)

    # load the model.
    print('[Info] Load the model.')

    if args.backbone == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes, atte=args.arch, ratio=args.reduction_ratio, dilation = args.dilation_value)
    elif args.backbone == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes, atte=args.arch, ratio=args.reduction_ratio, dilation = args.dilation_value)
    elif args.backbone == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes, atte=args.arch, ratio=args.reduction_ratio, dilation = args.dilation_value)
    elif args.backbone == 'resnet101':
        model = resnet.resnet101(num_classes=num_classes, atte=args.arch, ratio=args.reduction_ratio, dilation = args.dilation_value)
    elif args.backbone == 'resnet152':
        model = resnet.resnet152(num_classes=num_classes, atte=args.arch, ratio=args.reduction_ratio, dilation = args.dilation_value)


    model = model.to(device)
    if use_multi_gpu : model = torch.nn.DataParallel(model)

    # define loss function.
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        # Load the checkpoint.
        print('[Info] Loading checkpoint.')
        if torch.cuda.device_count() > 1:
            checkpoint = torch.load(args.save_multi)
        else:
            checkpoint = torch.load(args.save)

        backbone = checkpoint['backbone']
        args.epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print('[Info] epoch {} backbone {}'.format(args.epoch, backbone))

    # run evaluate.
    if args.evaluate:
        _ = run_epoch(model, 'valid', [args.epoch, args.epoch], criterion, optimizer, valid_loader, valid_size, device)
        return

    # run train.
    best_acc1 = 0.
    for e in range(args.epoch, args.n_epochs + 1):
        adjust_learning_rate(optimizer, e, args)

        # train for one epoch
        _ = run_epoch(model, 'train', [e, args.n_epochs], criterion, optimizer, train_loader, train_size, device)

        # evaluate on validation set
        with torch.no_grad():
            acc1 = run_epoch(model, 'valid', [e, args.n_epochs], criterion, optimizer, valid_loader, valid_size, device)

        # Save checkpoint.
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': e,
            'backbone': args.backbone,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)

        if use_multi_gpu:
            save_checkpoint({
                'epoch': e,
                'backbone': args.backbone,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save_multi)

        print('[Info] acc1 {} best@acc1 {}'.format(acc1, best_acc1))

def run_epoch(model, mode, epoch, criterion, optimizer, data_loader, dataset_size, device):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    tq = tqdm(data_loader, desc='  - (' + mode + ')   ', leave=False)
    for data, target in tq:
        # prepare data
        data, target = data.to(device), target.to(device)

        # forward
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        if mode == 'train':
            # compte gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tq.set_description(' - ({}) [ epoch: {}/{} loss: {:.3f}/{:.3f} ] '.format(mode, epoch[0], epoch[1], losses.val, losses.avg))
        #tqdm.write
    tqdm.write(' - ({})  [ epoch: {}\ttop@1: {:.3f}\ttop@5: {:.3f}\tloss: {:.3f}\ttime: {:.3f}]'.format(mode, epoch, top1.avg, top5.avg, losses.avg, (time.time() - start)/60.))


    return top1.avg

def save_checkpoint(state, is_best, prefix):
    filename='checkpoints/{}_checkpoint.chkpt'.format(prefix)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/{}_best.chkpt'.format(prefix))
    print(' - [Info] The checkpoint file has been updated.')

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        bsz = target.size(0)
        '''
            https://pytorch.org/docs/stable/torch.html#torch.topk
            torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        '''
        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / bsz))
        return res

if __name__ == '__main__':
	main()
