import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


class FeatureDataset(Dataset):

    def __init__(self, pkl, domain_id, sudo_len, opt=None):
        idx = pkl['domain'] == domain_id
        self.data = pkl['data'][idx].astype(np.float32)
        self.label = pkl['label'][idx].astype(np.int64)
        self.domain = domain_id
        self.real_len = len(self.data)
        self.sudo_len = sudo_len

    def __getitem__(self, idx):
        idx %= self.real_len
        return self.data[idx], self.label[idx], self.domain

    def __len__(self):
        return self.sudo_len


class FeatureDataloader(DataLoader):

    def __init__(self, opt):
        self.opt = opt
        self.src_domain_idx = opt.src_domain_idx
        self.tgt_domain_idx = opt.tgt_domain_idx
        self.all_domain_idx = opt.all_domain_idx

        self.pkl = read_pickle(opt.data_path)
        sudo_len = 0
        for i in self.all_domain_idx:
            idx = self.pkl['domain'] == i
            sudo_len = max(sudo_len, idx.sum())
        self.sudo_len = sudo_len

        print("sudo len: {}".format(sudo_len))

        self.datasets = [
            FeatureDataset(
                self.pkl,
                domain_id=i,
                opt=opt,
                sudo_len=self.sudo_len,
            ) for i in self.all_domain_idx
        ]

        self.data_loader = [
            DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=opt.shuffle,
                # num_workers=2,
                num_workers=0,
                pin_memory=True,
                # drop last only for new picked data of compcar
                drop_last=True,
            ) for dataset in self.datasets
        ]

    def get_data(self):
        # this is return a iterator for the whole dataset
        return zip(*self.data_loader)
