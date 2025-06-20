# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:#args.warmup_epochs=40
        lr = args.lr * epoch / args.warmup_epochs #0.00025*epoch/40#在前40个epoch中，学习率从0线性增长到目标学习率
    else:#Cosine Annealing（余弦退火）
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:#差异化学习率 例如：backbone可能使用较小的学习率，而新增的分类头使用较大的学习率
        if "lr_scale" in param_group:#好像只有微调的代码会进这里
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
