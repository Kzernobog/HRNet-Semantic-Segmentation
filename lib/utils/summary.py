import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils.helpers import decode_seg_map_sequence,decode_segmap
import numpy as np
from tensorboardX.utils import figure_to_image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = None

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def visualize_image(self, writer, dataset, image, target, output, global_step,num_image=3):
        grid_image = make_grid(image[:num_image].clone().cpu().data, num_image, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:num_image], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), num_image, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:num_image], 1).detach().cpu().numpy(),
                                                       dataset=dataset), num_image, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)


    def vis_grid(self, dataset, image, target, pred, global_step, split):
        image = image.squeeze()
        image = np.transpose(image, axes=[1, 2, 0])
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0
        image = image.astype(np.uint8)

        target = target.squeeze()
        # seg_mask = target == 2
        target = decode_segmap(target, dataset=dataset)
        pred = pred.squeeze()
        pred = decode_segmap(pred, dataset=dataset).squeeze()

        fig = plt.figure(figsize=(7, 15), dpi=150)
        ax1 = fig.add_subplot(311)
        ax1.imshow(image)
        ax2 = fig.add_subplot(312)
        ax2.imshow(pred)
        ax3 = fig.add_subplot(313)
        ax3.imshow(target)
        self.writer.add_image(split, figure_to_image(fig), global_step)
        plt.clf()
