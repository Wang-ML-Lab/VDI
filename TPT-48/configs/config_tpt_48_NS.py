from easydict import EasyDict
import numpy as np
# set experiment configs
opt = EasyDict()
from time import localtime, strftime

# opt.data_src = "compcar_data/data/"
# opt.data_path = opt.data_src + "feature_resnet.pkl"
# # opt.data_path = opt.data_src + "feature_resnet18_new_data.pkl"

opt.normalize_domain = False

# now it is quarter
# opt.num_domain = 30
# opt.num_source = 1
# opt.num_target = opt.num_domain - opt.num_source
# opt.src_domain_idx = np.array([0, 12, 3, 4, 14, 8])  # tight_boundary
# opt.tgt_domain_idx = np.array([1, 2, 5, 6, 7, 9, 10, 11, 13])

opt.data_src = "train_data/data/month_temp_data.pkl"

# TODO: start from here
# finish the changing from compcar to tpt data dataset

# N->S
opt.src_domain = ['ND','VT','NH','ME', 'WA','MT','SD','MN','WI','MI','NY','MA','OR','ID','WY','NE','IA','IL', 'IN','OH', 'PA', 'NJ','CT','RI']
opt.tgt_domain = ['GA', 'OK', 'NC', 'SC', 'LA', 'KY', 'UT', 'MS', 'FL', 'MO', 'MD', 'DE', 'CO', 'CA', 'TN', 'TX', 'KS', 'AZ', 'NV', 'AL', 'VA', 'AR', 'WV', 'NM']

# opt.src_domain_idx = [0] # list(range(0,12))
# # opt.tgt_domain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]# list(range(12,30))
# opt.tgt_domain_idx = list(set(range(0, 30)) - set(opt.src_domain_idx))
opt.all_domain = opt.src_domain + opt.tgt_domain
opt.num_domain = len(opt.all_domain)
opt.state2num = dict(zip(opt.all_domain, range(opt.num_domain)))
# opt.num2state = dict(zip(range(opt.num_domain), opt.all_domain))

opt.src_domain_idx = [opt.state2num[i] for i in opt.src_domain]
opt.tgt_domain_idx = [opt.state2num[i] for i in opt.tgt_domain]


opt.num_source = len(opt.src_domain_idx)
opt.num_target = len(opt.tgt_domain_idx)
# opt.num_domain = opt.num_source + opt.num_target

opt.all_domain_idx = opt.src_domain_idx + opt.tgt_domain_idx

# wheather shuffle data
opt.shuffle = True


# opt.dataset = "data/toy_d15_spiral_tight_boundary.pkl"
# opt.dataset = "data/toy_d15_quarter_circle.pkl"
opt.model = "IDI"
opt.d_loss_type = "DANN_loss" # "CIDA_loss" # "GRDA_loss" 
# opt.model = "SO"
# opt.model = "DANN"
# opt.model = "CDANN"
# opt.model = 'ADDA'
# opt.model = 'MDD'
print("model: {}".format(opt.model))

opt.use_pretrain_R = True
opt.pretrain_R_path = "pretrain_weight/netR_8_dann_tpt.pth"
opt.pretrain_U_path = "pretrain_weight/netU_8_dann_tpt.pth"

# opt.pretrain_R_path = "data/netR_8_dann_compcar_less_data.pth"
# opt.pretrain_U_path = "data/netU_8_dann_compcar_less_data.pth"
# opt.pretrain_R_path =  "data/netR_4_dann_compcar_8_dim.pth" # "data/netR_2_dann.pth" # "data/netR_4_dann.pth"
# opt.pretrain_U_path = "data/netU_4_dann_compcar_8_dim.pth" # "data/netU_2_dann.pth" # "data/netU_4_dann.pth"
opt.fix_u_r = False


opt.lambda_gan = 0.3 # 0.7 # 0.5 # 0.5
opt.lambda_reconstruct = 10 # 10 # 500
opt.lambda_u_concentrate = 0.8 # 0.8
opt.lambda_beta = 0.8

opt.lambda_beta_alpha = 0.6 # 0.1

# for exponential lr
opt.lr_e = 1 * 1e-4
opt.lr_d = 1 * 1e-4


# for warm up
opt.init_lr = 1e-6
opt.peak_lr_e = 3e-5 # 3.2*1e-4 # 1 * 1e-4 # 1.1 * 1e-4 # 2e-4 
opt.peak_lr_d = 3e-5 # 3.2*1e-4 # 1 * 1e-4 # 1.1 * 1e-4 # 2e-4 
opt.final_lr = 1e-8
opt.warmup_steps = 20 # 70


opt.seed = 2333 # 123 # 1 
opt.num_epoch = 500 # 500
opt.batch_size = 16 # 10

opt.use_visdom = False
opt.visdom_port = 2002
opt.test_on_all_dmn = True
# opt.outf = "dump"
tmp_time = localtime()
opt.outf = "result_save/{}".format(strftime("%Y-%m-%d %H:%M:%S", tmp_time))
opt.outf_vis = "result_save/{}".format(strftime("%Y-%m-%d\ %H\:%M\:%S", tmp_time))

opt.save_interval = 100
opt.test_interval = 20 # 10  # 20

opt.device = "cuda"
opt.gpu_device = "2"
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
# opt.wgan = False  # do not use wgan to train
opt.no_bn = True  # do not use batch normalization

# network parameter
opt.num_hidden = 512 # 512 # 512
# opt.num_class = 4  # how many classes for classification input data x
opt.seq_len = 6
opt.input_dim = 6  # the dimension of input data x
opt.group_len = 12
opt.bound_prediction = False

# we will solve the 1 dimension first
opt.u_dim = 8 # 4 # 2  # the dimension of local domain index u
opt.beta_dim = 2

# GMM model
opt.num_components = 5  # number of mixture of gaussians
# prior
opt.prior_pi = np.ones(opt.num_components) / opt.num_components
opt.prior_mean = np.linspace(-1, 1, num=opt.num_components)
opt.prior_std = 1

# for grda discriminator
opt.sample_v = 27

# for freeze u net
opt.freeze_u_epoch = -1 # set -1 to not freeze u net

opt.fixed_var = 1e-2

opt.sigma_beta = 1e-2

# how many nodes to save
opt.save_sample = 100 

# # num of samples for each class
# opt.sample_per_class = np.array([1163, 6136, 11656, 5196])
