from easydict import EasyDict
import numpy as np
from time import localtime, strftime
# set experiment configs
opt = EasyDict()

opt.num_domain = 30
opt.num_source = 6
opt.num_target = opt.num_domain - opt.num_source
opt.src_domain_idx = np.arange(opt.num_source).tolist()
opt.tgt_domain_idx = np.arange(opt.num_source, opt.num_domain).tolist()

opt.dataset = "data/toy_d15_quarter_circle.pkl"

opt.lambda_gan = 0.2
opt.lambda_reconstruct = 300
opt.lambda_u_concentrate = 0.1
opt.lambda_beta = 0.1
opt.lambda_beta_alpha = 0.1

opt.use_pretrain_R =  False
# opt.pretrain_R_path =  "data/netR_1_c30.pth" # "data/netR_2_dann.pth" # "data/netR_4_dann.pth"
# opt.pretrain_U_path = "data/netU_1_c30.pth" # "data/netU_2_dann.pth" # "data/netU_4_dann.pth"
opt.fix_u_r = False

opt.d_loss_type = "DANN_loss_mean" # "CIDA_loss" # "GRDA_loss" # "DANN_loss"

# for warm up
opt.init_lr = 1e-6
opt.peak_lr_d = 1.1 * 1e-4
opt.peak_lr_e = 1.1 * 1e-4
opt.final_lr = 1e-8
opt.warmup_steps = 40


opt.seed = 2333
opt.num_epoch = 800
opt.batch_size = 16

opt.use_visdom = False # True
opt.visdom_port = 2000
opt.test_on_all_dmn = False
tmp_time = localtime()
opt.outf = "result_save/{}".format(strftime("%Y-%m-%d %H:%M:%S", tmp_time))

opt.save_interval = 50
opt.test_interval = 20  # 20

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

opt.u_dim = 2      # the dimension of local domain index u
opt.beta_dim = 2   # the dimension of global domain index beta
