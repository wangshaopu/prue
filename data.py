import os
from multiprocessing import cpu_count

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def __get_dataloader(args, train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader


def get_tiny_imagenet(args):
    data_dir = os.path.join(args.data, 'tiny-200')
    normalize = transforms.Normalize((0.480, 0.448, 0.397),
                                     (0.276, 0.269, 0.282))

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'validation'), transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    return __get_dataloader(args, train_dataset, val_dataset)


def get_imagenet(args):
    data_dir = os.path.join(args.data, 'ILSVRC2012')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'validation'), transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return __get_dataloader(args, train_dataset, val_dataset)


def get_CIFAR(args, num_classes=10):
    data_root_dir = args.data

    if num_classes == 10:
        dataset = datasets.CIFAR10
    elif num_classes == 100:
        dataset = datasets.CIFAR100

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_dataset = dataset(
        data_root_dir, download=False, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = dataset(
        data_root_dir, download=False, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    return __get_dataloader(args, train_dataset, val_dataset)


def __get_prune_dataloader(args, dataset):
    # batch_size = int(len(dataset.targets) / max(dataset.targets))
    # Depending on your GPU memory capacity, the larger the better.
    batch_size = 500
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False)
    return train_loader


def get_cifar_prune_loader(args, num_classes=10):
    data_root_dir = args.data

    if num_classes == 10:
        dataset = datasets.CIFAR10
    elif num_classes == 100:
        dataset = datasets.CIFAR100

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    dataset = dataset(
        data_root_dir, download=False, train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset.targets, indices = torch.sort(torch.tensor(dataset.targets))
    dataset.data = dataset.data[indices]

    return __get_prune_dataloader(args, dataset)


def get_tiny_prune_loader(args):
    data_dir = os.path.join(args.data, 'tiny-200')
    normalize = transforms.Normalize((0.480, 0.448, 0.397),
                                     (0.276, 0.269, 0.282))

    dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    return __get_prune_dataloader(args, dataset)


def get_imagenet_prune_loader(args):
    data_dir = os.path.join(args.data, 'ILSVRC2012')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return __get_prune_dataloader(args, dataset)
