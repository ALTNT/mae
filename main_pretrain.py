# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm #timm提供了大量的预训练模型，可以直接用于各种计算机视觉任务，如图像分类、目标检测、语义分割等‌

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)#这个函数不是系统库，是自己写的 因为只有一张卡所以这里只执行了：在分布式训练中只让主进程（master process）输出日志信息 ，避免多个进程同时打印造成日志混乱

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)#device(type='cuda')

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()#0
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic 随机裁剪输入图像并缩放到指定尺寸（args.input_size），裁剪区域的比例范围在原始图像的20%~100%之间（scale=(0.2, 1.0)），使用双三次插值（interpolation=3）保持图像质量。
            transforms.RandomHorizontalFlip(),#以50%概率水平翻转图像，简单高效地扩充数据多样性，模拟镜像视角
            transforms.ToTensor(),#将PIL图像或NumPy数组转换为PyTorch张量（Tensor），并自动将像素值从[0,255]归一化到[0,1]范围，同时调整维度顺序为[C,H,W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])#对张量进行标准化处理，使用ImageNet数据集的均值和标准差（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]），将数据分布调整到零均值和单位方差，加速模型收敛
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)#'/datasets01/imagenet_full_size/061417/train'
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()#1 当前参与训练的总进程数
        global_rank = misc.get_rank()#0 当前进程的唯一标识符（排名）
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)#主进程进这里
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)#mae_vit_base_patch16_dec512d8b norm_pix_loss=False

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()#64*1*1
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256 # 0.001 * 64 / 256 = 0.00025

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))#1.00e-03
    print("actual lr: %.2e" % args.lr)#2.50e-04

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:#False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers 这表明代码遵循了timm库的做法，对bias（偏置）和norm layers（归一化层）设置权重衰减为0
    #对模型参数进行分为两组
    # 1、需要权重衰减的参数 ：通常是卷积层和全连接层的权重，
    # 2、不需要权重衰减的参数 ：bias参数和归一化层参数（如BatchNorm、LayerNorm的权重和偏置）
    # 权重衰减（Weight Decay）是一种正则化技术，通过在损失函数中添加权重的L2范数来防止过拟合
#     对于bias和归一化层参数，通常不应用权重衰减，因为：
# - Bias参数 ：主要用于调整输出的偏移，不会导致模型复杂度增加
# - 归一化层参数 ：如BatchNorm、LayerNorm的scale和shift参数，它们有特殊的作用机制
### 为什么要区别对待不同参数？
# 权重衰减的作用 ：

# - 权重衰减（Weight Decay）是一种正则化技术，通过在损失函数中添加权重的L2范数来防止过拟合
# - 对于bias和归一化层参数，通常不应用权重衰减，因为：
#   - Bias参数 ：主要用于调整输出的偏移，不会导致模型复杂度增加
#   - 归一化层参数 ：如BatchNorm、LayerNorm的scale和shift参数，它们有特殊的作用机制
# 实际效果 ：

# - 这种差异化的权重衰减策略可以提高模型的训练效果和泛化能力
# - 是现代深度学习训练中的标准做法，特别是在Vision Transformer等模型中
# 这段代码体现了MAE项目在优化器设置上的精细化处理，遵循了当前最佳实践。
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))#优化器创建
    print(optimizer)
    loss_scaler = NativeScaler()#创建混合精度训练的损失缩放器，用于防止梯度下溢

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):#args.epochs=400
        if args.distributed:#False
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# python main_pretrain.py --model mae_vit_base_patch16 --batch_size 64
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
