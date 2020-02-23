import torch
import functools
from .deeplab_sbn.batchnorm import SynchronizedBatchNorm1d
from .deeplab_sbn.batchnorm import SynchronizedBatchNorm2d
from .deeplab_sbn.batchnorm import SynchronizedBatchNorm3d
import torch.nn as nn

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    BatchNorm2d_class = BatchNorm2d = SynchronizedBatchNorm2d
    # BatchNorm2d_class = BatchNorm2d = nn.BatchNorm2d
    relu_inplace = True
