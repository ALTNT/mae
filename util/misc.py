# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf


class SmoothedValue(object):#用于 滑动窗口统计 和 全局统计 的 SmoothedValue 类
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):#维护最近20个值的窗口 统计信息 ：提供中位数、最大值、当前值等 全局平均 ：计算所有历史值的平均
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():#True
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):#用于 训练过程中指标记录和日志输出 的 MetricLogger 类
    def __init__(self, delimiter="\t"):#用于跟踪和显示训练过程中的各种指标（如损失、准确率等）。 
        self.meters = defaultdict(SmoothedValue)#meters ：存储各种指标的字典，每个指标都是一个 SmoothedValue 对象   依赖的 SmoothedValue 类 
        self.delimiter = delimiter#输出时的分隔符，默认为制表符

    def update(self, **kwargs):#接受任意数量的关键字参数 自动处理PyTorch张量，提取标量值  为每个指标创建或更新对应的 SmoothedValue
        for k, v in kwargs.items():#{'loss': 2.1465320587158203}
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):#允许直接通过属性名访问指标，如 logger.loss 等
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):#输出格式 ： loss: 0.5234 (0.5123)    acc: 0.8567 (0.8456)
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter#{'lr': <util.misc.SmoothedValue object at 0x7eff3c0cf400>}

# metric_logger.log_every(data_loader, print_freq, header) print_freq=20 header='Epoch: [0]'
    def log_every(self, iterable, print_freq, header=None):#核心方法：log_every  这是最重要的方法，用于在训练循环中定期输出日志：
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        # 初始化计时器
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # 构建日志格式
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')#'Epoch: [0]  [{0:3d}/{1}]  eta: {eta}  {meters}  time: {time}  data: {data}  max mem: {memory:.0f}'
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):#在分布式训练中只让主进程（master process）输出日志信息 ，避免多个进程同时打印造成日志混乱
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print# 保存原始的print函数

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)# 检查是否强制打印
        force = force or (get_world_size() > 8)# 当进程数>8时也强制打印
        if is_master or force:# 只有主进程或强制打印时才执行
            now = datetime.datetime.now().time()# 添加时间戳
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)# 执行实际打印

    builtins.print = print# 替换全局的print函数


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:#False
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:#False
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:#False
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# 这个类是**混合精度训练（Automatic Mixed Precision, AMP）**的核心组件，
# (前向传播：使用FP16计算，可以节省内存和加速)
# 损失缩放 ：将损失乘以大数值（如2^16），防止梯度下溢
# 反向传播 ：计算缩放后的梯度
# 梯度恢复 ：将梯度除以缩放因子，恢复真实值
# 参数更新 ：使用恢复后的梯度更新参数
# 主要用于：

# 1. 防止梯度下溢 ：在FP16训练中，梯度值可能非常小，导致下溢为0
# 2. 梯度裁剪 ：防止梯度爆炸(通过设置最大范数限制梯度大小)
# 3. 梯度范数计算 ：监控训练过程中的梯度状态
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()#创建PyTorch的自动混合精度梯度缩放器 用于自动调整损失缩放因子

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)#缩放损失并反向传播 ---将损失乘以缩放因子，防止梯度下溢 执行反向传播计算梯度
        if update_grad:
            if clip_grad is not None:
                # 有梯度裁剪的情况
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place # 恢复梯度的真实值
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)# 梯度裁剪
            else:#True
                # 无梯度裁剪的情况
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)# 计算梯度范数
            self._scaler.step(optimizer)## 执行优化器步骤
            self._scaler.update()# 更新缩放因子
        else:
            norm = None
        return norm

    def state_dict(self):#检查点（checkpoint）保存
        return self._scaler.state_dict()#主要包括当前的缩放因子、增长因子、回退计数器等AMP相关参数

    def load_state_dict(self, state_dict):#检查点（checkpoint）加载
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)#PosixPath('output_dir')
    epoch_name = str(epoch)#
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):#计算所有进程间某个值的平均值
    world_size = get_world_size()#1 获取分布式训练中的总进程数
    if world_size > 1:# 如果是多进程分布式训练
        x_reduce = torch.tensor(x).cuda()## 将输入转换为GPU上的tensor
        dist.all_reduce(x_reduce)# 在所有进程间进行求和操作
        x_reduce /= world_size# 除以进程数得到平均值
        return x_reduce.item()# 返回标量值
    else:# 如果是单进程训练
        return x