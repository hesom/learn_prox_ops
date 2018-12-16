import cv2
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch_networks import DnCNN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def normalize(data):
    return data/255.

class PytorchDeployer(object):
    """Deploy a trained network."""

    def __init__(self, layers=17, modelPath="logs/net.pth", cuda=True):
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.layers = layers
        self.modelPath = modelPath
        net = DnCNN(channels=1, num_of_layers=layers)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        self.model = model.to(device=self.device)

    def deploy(self, images):
        """
        Deploy and denoise an array of images.

        :param images: Input images
        :type images: List of np.ndarray

        :returns: denoised image of the network
        :rtype: ndarray
        """
        images = torch.Tensor(images).to(device=self.device)
        with torch.no_grad():
            return torch.clamp(self.model(images), 0., 1.).cpu().numpy()