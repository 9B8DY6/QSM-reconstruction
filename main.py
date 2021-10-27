from torch.utils.data import DataLoader
from dataloader import QSMDataset
from options import Options
from model import DIP_QSM

opt = Options().parse()

dataset = QSMDataset(opt)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
dataloader.flist = dataset.flist

model = DIP_QSM(opt)

model.test(dataloader)
