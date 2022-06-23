from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from trainModule import *
from utils import experiment_setting, model_prune, get_prune_loader


def main(args):
    seed_everything(args.seed)
    dict_args = vars(args)

    model, train_loader, val_loader = experiment_setting(args)
    foldername = f'{args.arch}_{args.smooth}'

    if args.pruner:
        foldername = f'{args.arch}_{args.pruner}_{args.ratio}'
        if args.pruner == 'prue':
            # If use prue, need to sort the data in the loader
            prune_loader = get_prune_loader(args)
            model_prune(model, args, prune_loader)
        else:
            model_prune(model, args, train_loader)

    if args.smooth == 1:
        # Use label smoothing
        litmodule = SmoothModule(model, **dict_args)
    else:
        # Normal training
        litmodule = TrainingModule(model, **dict_args)
    checkpoint_callback = [
        ModelCheckpoint(
            dirpath='./',
            monitor='valid_acc',
            mode='max',
            filename=f'{foldername}_best',
        ),
        TQDMProgressBar(refresh_rate=60),
    ]
    # If you want to use more than one GPU, please refer to https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html
    trainer = pl.Trainer(benchmark=True,
                         precision=16,
                         gpus=[args.gpu],
                         default_root_dir=foldername,
                         max_epochs=args.epochs,
                         callbacks=checkpoint_callback
                         )
    trainer.fit(litmodule, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-a', '--arch', default='resnet20')
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument(
        '--data', default='/dataset', help='Path to datasets')
    parser.add_argument('--resume', default=None,
                        help='Where to load checkpoints')
    parser.add_argument('--dataset', default='cifar10',
                        help='cifar10 cifar100 tinyimagenet imagenet')

    parser.add_argument('--filename', default='resnet20_0_best.ckpt',
                        help='Need a dense model before pruning')
    parser.add_argument('--ratio', default=0.5, type=float,
                        help='The target sparsity ratio')
    parser.add_argument('--pruner', default=None, type=str,
                        help='snip magnitude random prue')

    parser.add_argument('--seed', default=87656, type=int)
    parser.add_argument('--smooth', default=0, type=int)

    args = parser.parse_args()
    main(args)
