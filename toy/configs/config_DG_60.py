from easydict import EasyDict
import numpy as np
# set experiment configs
opt = EasyDict()

# now it is quarter
opt.num_domain = 60
opt.num_source = 6
opt.num_target = opt.num_domain - opt.num_source
opt.src_domain_idx = [40,32,12,29,22,52]  # tight_boundary
opt.tgt_domain_idx = list(set(range(0, opt.num_domain)) - set(opt.src_domain_idx))

opt.dataset = "data/toy_d60_pi_random.pkl"
opt.d_loss_type = "GRDA_loss" # "DANN_loss" # "CIDA_loss" # "DANN_loss_mean"

opt.use_pretrain_R = True
opt.pretrain_R_path = "data/netR_4_dann_60_pi.pth" # "data/netR_4_dann.pth"
opt.pretrain_U_path = "data/netU_4_dann_60_pi.pth" # "data/netU_4_dann.pth"
opt.fix_u_r = False

opt.lambda_gan = 0.35
opt.lambda_reconstruct = 7 
opt.lambda_u_concentrate = 0.4
opt.lambda_beta = 0.8
opt.lambda_beta_alpha = 0.1

# for warm up
opt.init_lr = 3e-7
opt.peak_lr_e = 1e-4
opt.peak_lr_d = 1e-4
opt.final_lr = 1e-8
opt.warmup_steps = 40

opt.seed = 2333 
opt.num_epoch = 500
opt.batch_size = 32

opt.use_visdom = False # True
opt.visdom_port = 2000
opt.test_on_all_dmn = False
opt.outf = "dump"

opt.save_interval = 50
opt.test_interval = 10  # 20

opt.device = "cuda"
opt.gpu_device = "0"
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4
opt.no_bn = True   # do not use batch normalization
opt.normalize_domain = False

# network parameter
opt.num_hidden = 512
opt.num_class = 2  # how many classes for classification input data x
opt.input_dim = 2  # the dimension of input data x

opt.u_dim = 4      # the dimension of local domain index u
opt.beta_dim = 2   # the dimension of global domain index beta

# for grda discriminator
opt.sample_v = 30
