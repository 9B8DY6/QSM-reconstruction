import torch
import torch.nn as nn
import numpy as np


def TVLoss(x):
    x_cen = x[:, :, 1:-1, 1:-1, 1:-1]
    x_shape = x.shape
    grad_x = torch.zeros_like(x_cen)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:, :, i + 1:i + x_shape[2] - 1, j + 1:j + x_shape[3] - 1, k + 1:k + x_shape[4] - 1]
                if i * i + j * j + k * k == 0:
                    temp = torch.zeros_like(x_cen)
                else:
                    temp = (1.0 / np.sqrt(i * i + j * j + k * k)) * (x_slice - x_cen)
                grad_x = grad_x + temp

    return torch.mean(torch.abs(grad_x))
