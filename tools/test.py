# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from utils.summary import TensorboardSummary
from utils.metrics2 import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')


    # tensorboard directory
    writer_dict = {'writer' : TensorboardSummary(tb_log_dir)}
    writer_dict['writer'].create_summary()
    # evaluator
    evaluator = Evaluator(config.DATASET.NUM_CLASSES)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        print(module)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    # dump_input = torch.rand(
    #     (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'best.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    checkpoint = torch.load(model_state_file)
    # pretrained_dict = torch.load(model_state_file)
    # model_dict = model.state_dict()
    # assert set(k[6:] for k in pretrained_dict) == set(model_dict)
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
    #                     if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    # model_dict.update(pretrained_dict)
    model.load_state_dict(checkpoint['state_dict'])


    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    val_global_step = 0
    epoch = 0
    start = timeit.default_timer()
    # if 'val' in config.DATASET.TEST_SET:
        # mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
        #                                                    test_dataset, 
        #                                                    testloader, 
        #                                                    model)
    mean_IoU, IoU_array, idr_avg, mean_acc, val_global_step = testval(config, \
                    test_dataset, testloader, model, evaluator, writer_dict, val_global_step, epoch)
    
        # msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
        #     Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
        #     pixel_acc, mean_acc)
        # logging.info(msg)
        # logging.info(IoU_array)
    # elif 'test' in config.DATASET.TEST_SET:
    #     test(config, 
    #          test_dataset, 
    #          testloader, 
    #          model,
    #          sv_dir=final_output_dir)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
