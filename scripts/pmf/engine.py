import math
import sys
import time
import warnings
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from timm.data import Mixup
from timm.utils import ModelEma

import utils.deit_util as utils
from utils import AverageMeter, to_device


def train_one_epoch(data_loader: Iterable,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    loss_scaler=None,
                    fp16: bool = False,
                    max_norm: float = 0,  # clip_grad
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    writer: Optional[SummaryWriter] = None,
                    set_training_mode=True,
                    k_closest=2):
    global_step = epoch * len(data_loader)
    print(len(data_loader))

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model.train(set_training_mode)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = to_device(batch, device)
        support_class, support_tensor, support_labels, x_class, x, y = batch

        SupportTensor, SupportLabel, x, y = model.get_k_closest(x_class, support_tensor, support_labels,
                                                                x_class, x, y, k=k_closest)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        # forward
        with torch.cuda.amp.autocast(fp16):
            output = model(SupportTensor, SupportLabel, x)
        output = output.view(x.shape[0] * x.shape[1], -1).float()
        print(output)
        y = y.view(x.shape[0] * x.shape[1], -1).float()
        print(y)
        loss = criterion(output, y)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if fp16:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        metric_logger.update(n_ways=SupportLabel.max() + 1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

        # tensorboard
        if utils.is_main_process() and global_step % print_freq == 0:
            writer.add_scalar("train/loss", scalar_value=loss_value, global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=lr, global_step=global_step)

        global_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(data_loaders, model, criterion, device, seed=None, ep=None, k_closest=2):
    if isinstance(data_loaders, dict):
        test_stats_lst = {}
        test_stats_glb = {}

        for j, (source, data_loader) in enumerate(data_loaders.items()):
            print(f'* Evaluating {source}:')
            seed_j = seed + j if seed else None
            test_stats = _evaluate(data_loader, model, criterion, device, seed_j)
            test_stats_lst[source] = test_stats
            test_stats_glb[source] = test_stats['acc1']

        # apart from individual's acc1, accumulate metrics over all domains to compute mean
        for k in test_stats_lst[source].keys():
            test_stats_glb[k] = torch.tensor([test_stats[k] for test_stats in test_stats_lst.values()]).mean().item()

        return test_stats_glb
    elif isinstance(data_loaders, torch.utils.data.DataLoader):  # when args.eval = True
        return _evaluate(data_loaders, model, criterion, device, seed, ep, k_closest)
    else:
        warnings.warn(f'The structure of {data_loaders} is not recognizable.')
        return _evaluate(data_loaders, model, criterion, device, seed, k_closest)


@torch.no_grad()
def accuracy(output, original_label, treshold=0.5):
    """
    if above treshold, return top-1 accuracy, if under treshold, class should be 0.
    Could be improved, but works for now
    """
    values, argmax = torch.max(output, dim=1)
    class_indexes = values >= treshold
    non_class_indexes = ~class_indexes
    vals, amax = torch.max(original_label, dim=1)
    original_labels = amax
    original_labels[vals == 0] = -1

    ## Argmax for labels above treshold, zero for labels below threshold
    argmax_acc = torch.count_nonzero(argmax[class_indexes] == original_labels[class_indexes])
    zeros_acc = torch.count_nonzero(original_labels[non_class_indexes] == -1)
    acc = (argmax_acc + zeros_acc) / len(original_labels)

    return acc


@torch.no_grad()
def _evaluate(data_loader, model, criterion, device, seed=None, ep=None, k_closest=2):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('n_ways', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('n_imgs', utils.SmoothedValue(window_size=1, fmt='{value:d}'))
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=len(data_loader.dataset)))
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    if seed is not None:
        data_loader.generator.manual_seed(seed)
    for ii, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if ep is not None:
            if ii > ep:
                break

        batch = to_device(batch, device)  # Put device somewhere else as we have to load only a fraction on CPU
        support_class, support_tensor, support_labels, x_class, x, y = batch

        # compute output
        with torch.cuda.amp.autocast():
            # Make K an argument here, fix x class issues
            SupportTensor, SupportLabel, x, y = model.get_k_closest(support_class, support_tensor, support_labels,
                                                                    x_class, x, y, k=k_closest)
            output = model(SupportTensor, SupportLabel, x)

        output = output.view(x.shape[0] * x.shape[1], -1).float()
        y = y.view(x.shape[0] * x.shape[1], -1).float()

        loss = criterion(output, y)
        acc1 = accuracy(output, y)
        acc5 = torch.tensor(1)

        batch_size = x.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.update(n_ways=SupportLabel.max() + 1)
        metric_logger.update(n_imgs=SupportTensor.shape[1] + x.shape[1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std

    return ret_dict
