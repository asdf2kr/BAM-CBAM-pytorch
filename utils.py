import os
import torch
import torchvision
import torchvision.transforms as transforms
# https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset

def prepare_dataloaders(args):
    '''
        ImageNET datasets.
        pytorch.org/docs/stable/torchvision/datasets.html#imagenet
    '''
    data_path = os.path.join(os.getcwd(), 'Datas')
    train_transform = transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.RandomCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)),
                           ])
    valid_transform = transform=transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)),
                           ])
    if args.datasets == 'cifar10':
        '''
        train_transform = transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225)),
                               ])
        valid_transform = transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225)),
                               ])
        '''
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
        valid_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=valid_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = args.batch,
                                                    num_workers = args.workers,
                                                    shuffle = True)

        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size = args.batch,
                                                    num_workers = args.workers)
    elif args.datasets == 'imagenet':
        # ImageNet 2012 Classification Dataset.
        train_dataset = torchvision.datasets.ImageNet(root=data_path, split='train', download=True, transform=train_transform)
        valid_dataset = torchvision.datasets.ImageNet(root=data_path, split='val', download=True, transform=valid_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = args.batch,
                                                    num_workers = args.workers,
                                                    shuffle = True)

        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size = args.batch,
                                                    num_workers = args.workers,
                                                    shuffle = True)

    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)
