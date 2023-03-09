import os
import torch
import numpy as np
import random
import pickle
import argparse
import importlib.util

# from configs.config_15 import opt
# from configs.config_30 import opt
# from configs.config_60 import opt
# from configs.config_15_random import opt
# from configs.config_60_random import opt
# from configs.config_60_random_pi import opt

# loading the config files
parser = argparse.ArgumentParser(description='Choose the configs to run.')
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

use_config_spec = importlib.util.spec_from_file_location(args.config,"configs/{}.py".format(args.config))
config_module = importlib.util.module_from_spec(use_config_spec)
use_config_spec.loader.exec_module(config_module)
opt = config_module.opt

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_device

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if opt.model == "IDI":
    from model.model import IDI as Model
elif opt.model == "DANN":
    from model.model import DANN as Model
elif opt.model == "CDANN":
    from model.model import CDANN as Model
    opt.cond_disc = True
elif opt.model == "ADDA":
    from model.model import ADDA as Model
elif opt.model == "MDD":
    from model.model import MDD as Model 
model = Model(opt).to(opt.device)

data_source = opt.dataset

# load the data
from dataset.dataset import *

data_source = opt.dataset

with open(data_source, "rb") as data_file:
    data_pkl = pickle.load(data_file)
print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

try:
    opt.angle = data_pkl['angle']
except:
    print("The dataset has no angle data.")

data = data_pkl['data']
data_mean = data.mean(0, keepdims=True)
data_std = data.std(0, keepdims=True)
data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
datasets = [ToyDataset(data_pkl, i, opt) for i in range(opt.num_domain)]  # sub dataset for each domain

dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
dataloader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=opt.batch_size
)

# train
for epoch in range(opt.num_epoch):
    if epoch == 0:
        model.test(epoch, dataloader)
    model.learn(epoch, dataloader)
    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:    
        model.test(epoch, dataloader)
        