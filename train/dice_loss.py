import torch
import numpy as np

###
# This code was taken from the article "Implementation of All Loss Functions - Deep Learning in Numpy, TensorFlow, and PyTorch"
# Author: Arjun Sarkar
# Source: https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning-in-numpy-tensorflow-and-pytorch-e20e72626ebd
###
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_true * y_pred, dim=(1,2,3))
        sum_of_squares_pred = torch.sum(torch.square(y_pred), dim=(1,2,3))
        sum_of_squares_true = torch.sum(torch.square(y_true), dim=(1,2,3))
        dice = 1 - (2 * intersection + self.smooth) / (sum_of_squares_pred + sum_of_squares_true + self.smooth)
        return dice
