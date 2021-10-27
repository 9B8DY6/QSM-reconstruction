import numpy as np
import torch
from os import listdir
from os.path import join
from scipy import io as sio
from torch.utils.data import Dataset


class QSMDataset(Dataset):
    def __init__(self, opt):
        self.data_root = opt.data_root

        self.flist = []
        for aFolder in sorted(listdir(self.data_root)):
            folder_root = join(self.data_root, aFolder)
            for aImg in sorted(listdir(folder_root)):
                self.flist.append(join(folder_root, aImg))

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        phase, mask, mag, voxel_size, B0 = self.read_mat(self.flist[idx])
        voxel_size = np.squeeze(voxel_size).astype(np.float32)
        voxel_size = voxel_size[[1, 0, 2]]
        B0 = np.squeeze(B0).astype(np.float32)
        B0 = B0[[1, 0, 2]]
        phase = phase * mask

        phase = torch.from_numpy(phase).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mag = torch.from_numpy(mag).unsqueeze(0)
        voxel_size = torch.from_numpy(voxel_size)
        B0 = torch.from_numpy(B0)

        return phase, mask, mag, voxel_size, B0

    def read_mat(self, filename):
        mat = sio.loadmat(filename, verify_compressed_data_integrity=False)
        data = mat['phase']
        mask = mat['mask']
        voxel_size = np.squeeze(mat['voxel_size'])
        B0_dir = np.squeeze(mat['B0_dir'])
        mag = mat['mag']
        mag = (mag - np.amin(mag)) / (np.amax(mag) - np.amin(mag))

        return data, mask, mag, voxel_size, B0_dir
