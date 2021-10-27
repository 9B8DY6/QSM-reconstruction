import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft3(data):
    assert data.size(-1) == 2
    data = torch.fft(data, 3, normalized=False)
    data = fftshift(data, dim=(-4, -3, -2))
    return data


def ifft3(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-4, -3, -2))
    data = torch.ifft(data, 3, normalized=False)
    return data


def to_complex(data, mode='zero'):
    data = data.unsqueeze(5)
    if mode == 'zero':
        out = torch.cat([data, torch.zeros_like(data)], dim=5)
    elif mode == 'tile':
        out = torch.cat([data, data], dim=5)
    else:
        raise NotImplementedError('to_complex [%s] is not implemented', mode)
    return out


def make_dipole_kernel(matrix_size, voxel_size, B0_dir):
    Y, X, Z = np.meshgrid(np.linspace(-matrix_size[0] / 2, matrix_size[0] / 2 - 1, matrix_size[0]),
                          np.linspace(-matrix_size[1] / 2, matrix_size[1] / 2 - 1, matrix_size[1]),
                          np.linspace(-matrix_size[2] / 2, matrix_size[2] / 2 - 1, matrix_size[2]))
    X = X / (matrix_size[1] * voxel_size[1])
    Y = Y / (matrix_size[0] * voxel_size[0])
    Z = Z / (matrix_size[2] * voxel_size[2])

    np.seterr(divide='ignore', invalid='ignore')
    D = 1 / 3 - np.divide(np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2]), np.square(X) + np.square(Y) + np.square(Z))
    D = np.where(np.isnan(D), 0, D)

    return D.astype(np.float32)


def pad(x, num_down=3):
    size = x.size()
    n = 2 ** num_down
    padY = (n - size[2] % n) % n
    padX = (n - size[3] % n) % n
    padZ = (n - size[4] % n) % n

    padY1 = int(padY / 2)
    padX1 = int(padX / 2)
    padZ1 = int(padZ / 2)

    padded = F.pad(x, [padZ1, padZ - padZ1, padX1, padX - padX1, padY1, padY - padY1])
    return padded, size[2], size[3], size[4], padY1, padX1, padZ1
