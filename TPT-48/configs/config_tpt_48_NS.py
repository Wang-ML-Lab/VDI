from easydict import EasyDict
from time import localtime, strftime
# set experiment configs
opt = EasyDict()

opt.data_src = "train_data/data/month_temp_data.pkl"

# N->S
opt.src_domain = [
    'ND', 'VT', 'NH', 'ME', 'WA', 'MT', 'SD', 'MN', 'WI', 'MI', 'NY', 'MA',
    'OR', 'ID', 'WY', 'NE', 'IA', 'IL', 'IN', 'OH', 'PA', 'NJ', 'CT', 'RI'
]
opt.tgt_domain = [
    'GA', 'OK', 'NC', 'SC', 'LA', 'KY', 'UT', 'MS', 'FL', 'MO', 'MD', 'DE',
    'CO', 'CA', 'TN', 'TX', 'KS', 'AZ', 'NV', 'AL', 'VA', 'AR', 'WV', 'NM'
]

opt.all_domain = opt.src_domain + opt.tgt_domain
opt.num_domain = len(opt.all_domain)
opt.state2num = dict(zip(opt.all_domain, range(opt.num_domain)))

opt.src_domain_idx = [opt.state2num[i] for i in opt.src_domain]
opt.tgt_domain_idx = [opt.state2num[i] for i in opt.tgt_domain]

opt.num_source = len(opt.src_domain_idx)
opt.num_target = len(opt.tgt_domain_idx)

opt.all_domain_idx = opt.src_domain_idx + opt.tgt_domain_idx

# wheather shuffle data
opt.shuffle = True
opt.d_loss_type = "DANN_loss"  # "CIDA_loss" # "GRDA_loss" # "DANN_loss_mean"
opt.use_pretrain_R = True
opt.pretrain_R_path = "pretrain_weight/netR_8_dann_tpt.pth"
opt.pretrain_U_path = "pretrain_weight/netU_8_dann_tpt.pth"

opt.fix_u_r = False

opt.lambda_gan = 0.3
opt.lambda_reconstruct = 10
opt.lambda_u_concentrate = 0.8
opt.lambda_beta = 0.8
opt.lambda_beta_alpha = 0.6

# for warm up
opt.init_lr = 1e-6
opt.peak_lr_e = 3e-5
opt.peak_lr_d = 3e-5
opt.final_lr = 1e-8
opt.warmup_steps = 20

opt.seed = 2333
opt.num_epoch = 500
opt.batch_size = 16

opt.use_visdom = False  # True
opt.visdom_port = 2000
tmp_time = localtime()
opt.outf = "result_save/{}".format(strftime("%Y-%m-%d %H:%M:%S", tmp_time))

opt.save_interval = 100
opt.test_interval = 20

opt.device = "cuda"
opt.gpu_device = "0"
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
opt.normalize_domain = False
opt.no_bn = True  # do not use batch normalization

# network parameter
opt.num_hidden = 512
opt.seq_len = 6
opt.input_dim = 6  # the dimension of input data x
opt.group_len = 12

opt.u_dim = 8  # the dimension of local domain index u
opt.beta_dim = 2

# for grda discriminator
opt.sample_v = 27

# how many nodes to save
opt.save_sample = 100
