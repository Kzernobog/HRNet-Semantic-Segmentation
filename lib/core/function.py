# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time
import sys

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.helpers import calculate_weights_batch
import utils.distributed as dist


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, criterion, evaluator,
          writer_dict):
    # Training
    model.train()

    num_img_iter = len(trainloader)


    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    t_train_bar = tqdm(trainloader, desc="Training => epoch: {}".format(epoch))

    # reset evaluator
    evaluator.reset()

    for i_iter, batch in enumerate(t_train_bar):


        images, context, labels, _, _ = batch
        size = labels.size()
        images = images.cuda()
        context = context.cuda()
        labels = labels.long().cuda()
        images = torch.cat((images, context), 1)


        # losses, _ = model(images, labels)
        output = model(images)
        class_weights = torch.from_numpy(calculate_weights_batch(labels,
                                                                 config.DATASET.NUM_CLASSES).astype(np.float32)).cuda()
        loss = torch.unsqueeze(criterion(output, labels, class_weights), 0).mean()
        # print('losses shape: ', losses.shape)
        # loss = losses.mean().contiguous()
        # print('loss shape: ', loss.shape)
        # print('loss: ', loss.item())

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # converting ouput to prediction
        pred_output = output[config.TEST.OUTPUT_INDEX]
        pred_output = F.interpolate(
            input=pred_output, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        pred = pred_output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        # retrieving images and labels to log on tensorboard
        images = images.clone().cpu().numpy()
        labels = labels.clone().cpu().numpy()

        # for logging purposes
        evaluator.add_batch(labels, pred)

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        writer.writer.add_scalar('loss/train_batch_loss', reduced_loss.item(), i_iter + (num_img_iter*epoch))
        t_train_bar.set_description('Training => Epoch [{}/{}] Epoch Loss: {}; Loss: {}'.format(epoch, num_epoch, ave_loss.average(), loss))
        # writer.vis_grid(config.DATASET.DATASET, images, labels, output, i_iter + (num_img_iter*epoch), 'train')

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            # logging.info(msg)

    # all tensorboard logging related stuff
    # idr thresholds
    idr_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65]

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    # class_iou = evaluator.class_IOU(class_num=2)
    recall,precision = evaluator.pdr_metric(class_id=2)
    FPR = evaluator.get_false_idr(class_id=2)
    iIoU = evaluator.get_instance_iou(threshold=0.2)
    idr_avg = np.array([evaluator.get_idr(class_value=2, threshold=value) for value in idr_thresholds])
    writer.writer.add_scalar('loss/train_epoch_loss', ave_loss.average(), epoch)
    writer.writer.add_scalar('metrics/train_miou', mIoU, epoch)
    # writer.writer.add_scalar('metrics/train_SO_iou', class_iou, epoch)
    writer.writer.add_scalar('metrics/train_acc', Acc, epoch)
    writer.writer.add_scalar('metrics/train_acc_cl', Acc_class, epoch)
    writer.writer.add_scalar('metrics/train_fwIoU', FWIoU, epoch)
    if recall is not None:
        writer.writer.add_scalar('metrics/train_pdr_epoch',recall,epoch)
    if precision is not None:
        writer.writer.add_scalar('metrics/train_precision_epoch',precision,epoch)
    if idr_avg is not None:
        writer.writer.add_scalar('metrics/train_idr_epoch', np.mean(idr_avg), epoch)

    # writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, criterion, writer_dict, epoch):
    model.eval()

    # reset the evaluator

    num_img_itr = len(testloader)

    # tensorboad logging
    writer = writer_dict['writer']

    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))

    t_val_bar = tqdm(testloader)

    count = 0

    with torch.no_grad():
        for idx, batch in enumerate(t_val_bar):
            if count > 100:
                model.eval()
            count += 1
            image, context, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            context = context.cuda()
            image = torch.cat((image, context), 1)



            # losses, pred = model(image, label)
            pred = model(image)
            class_weights = torch.from_numpy(calculate_weights_batch(label,
                                                                     config.DATASET.NUM_CLASSES).astype(np.float32)).cuda()
            # class_weights = calculate_weights_batch(labels, config.DATASET.NUM_CLASSES)
            losses = torch.unsqueeze(criterion(pred, label, class_weights), 0).mean()
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            # if idx % 10 == 0:
            #     print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss


            # update evaluator
            ave_loss.update(reduced_loss.item())
            t_val_bar.set_description('Validation => Epoch [{}/{}] Epoch Loss: {}; Loss: {}'.format(epoch, config.TRAIN.END_EPOCH, ave_loss.average(), loss))
            writer.writer.add_scalar('loss/val_batch_loss', reduced_loss.item(), idx + (num_img_itr)*epoch)

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        # if dist.get_rank() <= 0:
        #     logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))


    writer.writer.add_scalar('loss/val_epoch_loss', ave_loss.average(), epoch)


    # writer = writer_dict['writer']
    # global_steps = writer_dict['valid_global_steps']
    # writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    # writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    # writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model, evaluator, writer_dict,
            val_global_step, epoch, sv_dir='', sv_pred=False):
    confusion_matrix = np.zeros( (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    # reset the evaluator
    evaluator.reset()

    # initialise the tensorboard writer
    writer = writer_dict['writer']

    # fluff
    t_val_bar = tqdm(testloader)

    count = 0

    with torch.no_grad():
        for index, batch in enumerate(t_val_bar):
            if count > 50:
                model.eval()
            count += 1
            image, context, label, _, name = batch
            image_cm = torch.cat((image, context), 1)
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image_cm,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            # if sv_pred:
            #     sv_path = os.path.join(sv_dir, 'test_results')
            #     if not os.path.exists(sv_path):
            #         os.mkdir(sv_path)
            #     test_dataset.save_pred(pred, sv_path, name)


            # taking predictions
            pred = pred.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)


            # retrieving images and labels to log on tensorboard
            images = image.clone().cpu().numpy()
            label = label.clone().cpu().numpy()

            # for logging purposes
            evaluator.add_batch(label, pred)

            # fluff
            t_val_bar.set_description('Generating Metrics and Visualisations => Epoch [{}/{}]'.format(epoch, config.TRAIN.END_EPOCH))

            # log on tensorbaord
            writer.vis_grid(config.DATASET.DATASET, images[0], label[0], pred[0], val_global_step, 'val')
            val_global_step += 1

            if index % 100 == 0:
                # logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                # logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    # all tensorboard logging related stuff
    # idr thresholds for instance metric
    idr_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65]
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FPR = evaluator.get_false_idr(class_id=2)
    iIoU = evaluator.get_instance_iou(threshold=0.2)
    # class_iou = evaluator.class_IOU(class_num=2)
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    recall,precision=evaluator.pdr_metric(class_id=2)
    # idr = evaluator.get_idr(class_value=2)
    idr_avg = np.array([evaluator.get_idr(class_value=2, threshold=value) for value in idr_thresholds])
    writer.writer.add_scalar('metrics/val_idr_0.20', idr_avg[0], epoch)
    # writer.writer.add_scalar('metrics/val_SO_iou', class_iou, epoch)
    writer.writer.add_scalar('metrics/val_idr_0.65', idr_avg[-1], epoch)
    writer.writer.add_scalar('metrics/val_idr_avg', np.mean(idr_avg), epoch)
    writer.writer.add_scalar('metrics/val_miou', mIoU, epoch)
    writer.writer.add_scalar('metrics/val_acc', Acc, epoch)
    writer.writer.add_scalar('metrics/val_acc_cl', Acc_class, epoch)
    writer.writer.add_scalar('metrics/val_fwIoU', FWIoU, epoch)
    if recall is not None:
        writer.writer.add_scalar('metrics/val_pdr_epoch',recall,epoch)
    if precision is not None:
        writer.writer.add_scalar('metrics/val_precision_epoch',precision,epoch)
    if idr_avg is not None:
        writer.writer.add_scalar('metrics/val_idr_epoch', np.mean(idr_avg), epoch)

    logging.info('PDR: {}; FPR: {}; IDR_20: {}; iIoU: {}; mIoU'.format(recall, FPR, idr_avg[0], iIoU, mIoU))

    return mean_IoU, IoU_array, idr_avg[0], mean_acc, val_global_step


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
