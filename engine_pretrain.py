# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

# 预训练的训练循环

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
# 梯度累积是一种在内存或GPU数量有限的情况下，模拟更大批次大小训练的技术。通过累积多个小批次的梯度，然后一次性更新参数，可以达到与大批次训练相似的效果。
    accum_iter = args.accum_iter#梯度累积迭代次数，用于实现梯度累积（Gradient Accumulation）技术

    optimizer.zero_grad()# 初始化梯度，防止梯度累加

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):#torch.Size([64, 3, 224, 224])

        # we use a per iteration (instead of per epoch) lr scheduler# 只在累积周期开始时调整学习率
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)#torch.Size([64, 3, 224, 224])
# 前向传播计算损失
        with torch.cuda.amp.autocast():#PyTorch会自动将模型的前向传播从默认的 float32 精度转换为 float16 精度，以提高训练效率
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
# 将损失除以累积次数
        loss_value = loss.item()

        if not math.isfinite(loss_value):#False Return True if x is neither an infinity nor a NaN, and False otherwise.
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),## 反向传播，但只在累积周期结束时更新参数
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:# 只在累积周期结束时清零梯度
            optimizer.zero_grad()

        torch.cuda.synchronize()#用于同步 CPU 和 GPU 之间的操作。当调用这个函数时，它会阻塞 CPU 的执行，直到所有之前提交到 GPU 的任务（如内核启动、内存复制等）都完成为止。
# 然而，由于它的性能开销，应该谨慎使用，只在需要精确同步的场景下调用
        metric_logger.update(loss=loss_value)#{'loss': 2.1465320587158203}

        lr = optimizer.param_groups[0]["lr"]#0.0  ？这个是干什么的？
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)#loss_value=2.1465320587158203
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:#这里应该是用来画损失函数的曲线的
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)#0
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}