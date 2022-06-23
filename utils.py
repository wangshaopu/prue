import torch
import torch.nn.utils.prune as prune

import data
import models
import pruner
from trainModule import *

num_classes = {'cifar10': 10, 'cifar100': 100,
               'tinyimagenet': 200, 'imagenet': 1000}


def experiment_setting(args):
    '''
    All training hyperparameters are stored here for automatic loading. Forget about those long training scripts, all you need to do is to python it. 
    '''
    model = getattr(models, args.arch)(
        num_classes=num_classes[args.dataset])

    if args.dataset.startswith('cifar10'):
        args.lr = 0.1
        args.batch_size = 512
        args.milestones = [81, 122]
        args.epochs = 2
        args.gamma = 0.1
        args.weight_decay = 5e-4

        if args.dataset == 'cifar10':
            train_loader, val_loader = data.get_CIFAR(args, num_classes=10)
        else:
            train_loader, val_loader = data.get_CIFAR(args, num_classes=100)
    elif args.dataset == 'tinyimagenet':
        args.lr = 0.1
        args.batch_size = 256
        args.milestones = [60, 120, 160]
        args.epochs = 200
        args.gamma = 0.2
        args.weight_decay = 5e-4
        train_loader, val_loader = data.get_tiny_imagenet(args)
    elif args.dataset == 'imagenet':
        args.lr = 0.1
        args.batch_size = 256
        args.milestones = [30, 60]
        args.epochs = 90
        args.gamma = 0.1
        args.weight_decay = 1e-4
        train_loader, val_loader = data.get_imagenet(args)
    else:
        print('Unsupported Dataset.')

    return model, train_loader, val_loader


def get_prune_loader(args):
    '''
    PrUE-specific. Returns a data loader with a sorted dataset. 
    '''
    if args.dataset == 'cifar10':
        dataloader = data.get_cifar_prune_loader(args, num_classes=10)
    elif args.dataset == 'cifar100':
        dataloader = data.get_cifar_prune_loader(args, num_classes=100)
    elif args.dataset == 'tinyimagenet':
        dataloader = data.get_tiny_prune_loader(args)
    elif args.dataset == 'imagenet':
        dataloader = data.get_imagenet_prune_loader(args)
    else:
        print('Unsupported Dataset, Gong.')
    return dataloader


def model_prune(model, args, train_loader):
    '''
    Prune the model according the train_loader.
    '''
    TrainingModule.load_from_checkpoint(
        checkpoint_path=args.filename,
        model=model,
    )

    # ---Get ready for global pruning---
    module_to_prune = ()

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module_to_prune += ((module, 'weight'),)

    prune.global_unstructured(
        module_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )
    # ---------------------------------

    # Apply different pruning criteria
    if args.pruner == 'magnitude':
        prune.global_unstructured(
            module_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=args.ratio,
        )
    elif args.pruner == 'random':
        prune.global_unstructured(
            module_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=args.ratio,
        )
    else:
        getattr(pruner, args.pruner)(
            model, nn.CrossEntropyLoss(), train_loader, args.ratio, args.gpu)

    # Need a refresh to get the new mask to work
    prune.global_unstructured(
        module_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0,
    )
