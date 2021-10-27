import torch
import torch.nn as nn
import numpy as np
from os import makedirs
from os.path import join, isdir
from scipy import io as sio
from generator import Unet, PhysicsModel
from losses import TVLoss
from utils import init_net, pad, make_dipole_kernel
from tqdm import tqdm


class DIP_QSM:
    def __init__(self, opt):
        self.device = torch.device('cuda:' + str(opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
        self.lambda_tv = opt.lambda_tv
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.n_iters = opt.n_iters
        self.lr = opt.lr
        self.init_type = opt.init_type
        self.init_gain = opt.init_gain
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.opt = opt

        self.fidelity_loss = nn.L1Loss()

        self.G = Unet(1, 1, self.opt).to(self.device)
        self.PhysicsModel = PhysicsModel()

        self.G_optim = torch.optim.Adam(self.G.parameters(), self.lr, betas=(self.beta1, self.beta2))

    def test(self, dataloader):
        save_path = join(self.save_path, self.experiment_name, 'test_{}'.format(self.n_iters))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test.')
        for step, [phase, mask, _, voxel_size, B0] in enumerate(tqdm(dataloader)):
            init_net(self.G, self.init_type, self.init_gain)
            self.G.train()

            phase_pad, nY, nX, nZ, padY1, padX1, padZ1 = pad(phase, 3)
            mask_pad, _, _, _, _, _, _ = pad(mask, 3)
            dk = make_dipole_kernel([phase_pad.size(2), phase_pad.size(3), phase_pad.size(4)], np.squeeze(voxel_size.numpy()), np.squeeze(B0.numpy()))
            dk = torch.from_numpy(dk).unsqueeze(0).unsqueeze(1)

            phase_pad = phase_pad.to(self.device)
            dk = dk.to(self.device)
            mask_pad = mask_pad.to(self.device)

            for iter in tqdm(range(self.n_iters)):
                QSM = self.G(phase_pad) * mask_pad
                phase_recon = self.PhysicsModel(QSM, dk) * mask_pad

                fidelity_loss = self.fidelity_loss(phase_recon, phase_pad)
                tv_loss = TVLoss(QSM)
                total_loss = fidelity_loss + self.lambda_tv * tv_loss

                self.G_optim.zero_grad()
                total_loss.backward()
                self.G_optim.step()

            self.G.eval()
            QSM = np.squeeze((self.G(phase_pad) * mask_pad).to('cpu:0').detach().numpy())
            QSM = QSM[padY1:padY1 + nY, padX1:padX1 + nX, padZ1:padZ1 + nZ]

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            test_output = {'QSM': QSM}

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)
