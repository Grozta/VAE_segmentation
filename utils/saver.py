import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import logging
import numpy as np

def label2color(label,class_num=4):
    color_img = np.zeros((label.shape[0],label.shape[1],3))
    color_img[:,:,0][label==1]=1
    color_img[:,:,1][label==2]=1
    color_img[:,:,2][label==3]=1
    return color_img

class Saver():
    def __init__(self, display_dir,display_freq):
        self.display_dir = display_dir
        self.display_freq = display_freq
        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

  # write losses and images to tensorboard
    def write_display(self, total_it, loss, image=None, force_write=False):
        if force_write or (total_it + 1) % self.display_freq == 0:
            # write img
            if image is not None:
                for m in image:
                    image_dis = torchvision.utils.make_grid(image[m].detach().cpu(), nrow=5)/2 + 0.5
                    self.writer.add_image(m, image_dis, total_it)
            for l in loss:
                self.writer.add_scalar(l[0], l[1], total_it)
                print(l[0], l[1], total_it)
                
    def image_display(self,name, oringin_image, pred, groundTruth ,total_it,force_write=False):
        """oringin_image[8,4,16,256,256]"""
        if force_write or (total_it + 1) % self.display_freq == 0:
            # write img
            if oringin_image.is_cuda:
                oringin_image = oringin_image.detach().cpu()
            if pred.is_cuda:
                pred = pred.detach().cpu()
            if groundTruth.is_cuda:
                groundTruth = groundTruth.detach().cpu()
            oringin_image = np.argmax(oringin_image.numpy()[2,:,8,:,:],axis=0)
            pred = np.argmax(pred.numpy()[2,:,8,:,:],axis=0)
            groundTruth = np.argmax(groundTruth.numpy()[2,:,8,:,:],axis=0)
            img = np.array([label2color(oringin_image),label2color(pred),label2color(groundTruth)])
            self.writer.add_images(name, img, total_it,dataformats='NHWC')
