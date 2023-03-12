import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

from data_loader.utils import get_date_list


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


class WeatherDataLoader():

    def __init__(self, opt):
        self.opt = opt
        self.src_domain = opt.src_domain
        self.tgt_domain = opt.tgt_domain
        self.raw_data = read_pickle(opt.data_src)
        self.group_len = opt.group_len
        self.all_states = self.opt.all_domain
        self.sudo_len = max(
            [len(self.raw_data[state])
             for state in self.all_states]) - self.group_len + 1
        self.all_mean, self.all_std = self.__norm__()

        self.train_datasets = [
            CovidTrainDataset(self.raw_data[state],
                              isSrc=(state in self.src_domain),
                              all_mean=self.all_mean,
                              all_std=self.all_std,
                              domain_idx=self.opt.state2num[state],
                              sudo_len=self.sudo_len,
                              opt=opt) for state in self.all_states
        ]

        self.test_datasets = [
            CovidTestDataset(self.raw_data[state],
                             all_mean=self.all_mean,
                             all_std=self.all_std,
                             domain_idx=self.opt.state2num[state],
                             opt=opt) for state in self.all_states
        ]

        self.train_data_loader = [
            DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
            for dataset in self.train_datasets
        ]

        self.test_data_loader = [
            DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
            for dataset in self.test_datasets
        ]

    def __maxmin__(self):
        tmp_max = -1e9
        tmp_min = 1e9
        for state in self.all_states:
            tmp_max = max(tmp_max, max(self.raw_data[state]))
            tmp_min = min(tmp_min, min(self.raw_data[state]))

        return tmp_max, tmp_min

    def __norm__(self):
        seq_len = self.opt.seq_len
        all_data = np.array([])
        for state in self.src_domain:
            # be sure that we delete more data (the last few days)
            all_data = np.append(
                all_data, self.raw_data[state][:(self.sudo_len + seq_len - 1)])
        for state in self.tgt_domain:
            tmp_data = self.raw_data[state]
            group_len = self.group_len
            data_group_num = tmp_data.shape[0] // group_len
            for i in range(data_group_num):
                all_data = np.append(
                    all_data, tmp_data[i * group_len:i * group_len + seq_len])

        return all_data.mean(), all_data.std()

    def get_train_data(self):
        # this is return a iterator for the whole dataset
        return zip(*self.train_data_loader)

    def get_test_data(self):
        return zip(*self.test_data_loader)


class CovidTrainDataset(Dataset):

    def __init__(self, data, isSrc, all_mean, all_std, domain_idx, sudo_len,
                 opt):
        self.data = (data - all_mean) / all_std
        self.domain_idx = domain_idx
        self.isSrc = isSrc
        self.seq_len = opt.seq_len

        self.sudo_len = sudo_len
        self.opt = opt
        self.group_len = opt.group_len

        if isSrc:
            self.real_len = self.data.shape[0] - self.opt.group_len + 1
        else:
            # Assume that we have 244 days data, so doesn't handle many speciall cases!
            self.real_len = self.data.shape[0] // self.opt.group_len

    def __len__(self):
        return self.sudo_len

    def __getitem__(self, idx):
        if self.isSrc:
            x = self.data[idx:idx + self.seq_len]
            y = self.data[idx + self.seq_len:idx + self.seq_len * 2]
        else:
            if idx >= self.real_len:
                idx = idx % self.real_len

            x = self.data[idx * self.group_len:idx * self.group_len +
                          self.seq_len]
            y = self.data[idx * self.group_len +
                          self.seq_len:idx * self.group_len + 2 * self.seq_len]

        return x, y, idx, self.domain_idx


class CovidTestDataset(Dataset):

    def __init__(self, data, all_mean, all_std, domain_idx, opt):
        self.data = (data - all_mean) / all_std
        self.domain_idx = domain_idx
        self.seq_len = opt.seq_len
        self.opt = opt
        self.group_len = opt.group_len

        self.real_len = self.data.shape[0] // opt.group_len

    def __len__(self):
        return self.real_len

    def __getitem__(self, idx):
        x = self.data[idx * self.group_len:idx * self.group_len + self.seq_len]
        y = self.data[idx * self.group_len +
                      self.seq_len:idx * self.group_len + 2 * self.seq_len]

        return x, y, idx, self.domain_idx
