from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn.utils.prune as prune
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import models
from trainModule import *
from utils import experiment_setting, num_classes


def main(args):
    seed_everything(args.seed)
    dict_args = vars(args)

    model, train_loader, val_loader = experiment_setting(args)
    foldername = args.teacher_path.replace('.ckpt', '')
    foldername = f'{foldername}_to_{args.arch}_with_{args.temperature}'

    t_model = getattr(models, args.t_arch)(
        num_classes=num_classes[args.dataset])

    if args.pruned == 1:
        module_to_prune = ()

        for module in t_model.modules():
            if isinstance(module, nn.Conv2d):
                module_to_prune += ((module, 'weight'),)

        prune.global_unstructured(
            module_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0,
        )
    TrainingModule.load_from_checkpoint(
        args.teacher_path, model=t_model
    )

    litmodule = DistillModule(model, t_model, **dict_args)
    checkpoint_callback = [
        ModelCheckpoint(
            dirpath='./',
            monitor='valid_acc',
            mode='max',
            filename=f'{foldername}_best',
        ),
        TQDMProgressBar(refresh_rate=60),
    ]
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

    parser.add_argument('-a', '--arch', default='resnet8')
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--data', default='/dataset')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dataset', default='cifar10',
                        help='cifar10 cifar100 tinyimagenet imagenet')
    # 蒸馏用
    parser.add_argument('--temperature', default=4, type=int,
                        help='Distillation temperature')
    parser.add_argument('--pruned', default=1, type=int,
                        help='0 for dense teacher, 1 for sparse teacher')
    parser.add_argument('--t-arch', default='resnet20',
                        help='The teacher arch')
    parser.add_argument(
        '--teacher-path', default='resnet20_magnitude_0.5_best.ckpt', help='Path to the teacher model')

    parser.add_argument('--seed', default=87654, type=int)

    args = parser.parse_args()
    main(args)
