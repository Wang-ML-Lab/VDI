import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model.modules import *
import os
from visdom import Visdom
import pickle
import json

from model.lr_scheduler import TransformerLRScheduler
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from geomloss import SamplesLoss
# const
LARGE_NUM = 1e9


# ========================
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

# def get_weights_inverse_num_of_samples(no_of_class, samples_per_cls, power=1):
#     weights = 1.0 / np.array(np.power(samples_per_cls, power))
#     weights = weights / np.sum(weights) * no_of_class
#     return weights


# =========================
# the base model
class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)

        self.opt = opt
        self.device = opt.device
        self.use_visdom = opt.use_visdom
        if opt.use_visdom:
            self.env = Visdom(port=opt.visdom_port)
            self.test_pane = dict()
            self.test_pane_init = False

        # GMM prior
        self.prior_pi = opt.prior_pi
        self.prior_mean = opt.prior_mean
        self.prior_std = opt.prior_std
        self.prior_var = opt.prior_std**2

        self.num_domain = opt.num_domain
        if self.opt.test_on_all_dmn:
            self.test_dmn_num = self.num_domain
        else:
            self.test_dmn_num = self.opt.num_target

        self.outf = self.opt.outf
        self.train_log = self.outf + "/loss.log"
        self.model_path = self.outf + '/model.pth'
        self.out_pic_f = opt.outf + '/plt_pic'
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        if not os.path.exists(self.out_pic_f):
            os.mkdir(self.out_pic_f)
        with open(self.train_log, 'w') as f:
            f.write("log start!\n")

        # save all the config to json file
        with open("{}/config.json".format(self.outf), "w") as outfile:
            json.dump(self.opt, outfile, indent=2)


        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.src_domain_idx] = 1
        self.domain_mask = torch.IntTensor(mask_list).to(opt.device)

        # nan flag
        self.nan_flag = False

        # beta:
        self.use_beta_seq = None

        # save number
        # self.save_sample = self.opt.save_sample
        self.init_test = False

        self.group_len = opt.group_len
        self.seq_len = opt.seq_len

        # # weight for classfication
        # self.weight_for_nll = get_weights_inverse_num_of_samples(opt.num_class, opt.sample_per_class, 1)
        # self.weight_for_nll = torch.from_numpy(self.weight_for_nll).to(self.device).float()

    def learn(self, epoch, dataloader):
        self.train()

        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}
        self.new_u = []

        count = 0
        for data in dataloader.get_train_data():
            count += 1
            self.__set_input__(data)
            self.__train_forward__()
            new_loss_values = self.__optimize__()

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

            self.new_u.append(self.u_seq)

        self.new_u = self.my_cat(self.new_u)
        self.use_beta_seq = self.generate_beta(self.new_u)

        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
            print("epoch {}, loss: {}, lambda gan: {}".format(
                self.epoch, loss_values, self.opt.lambda_gan))

        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        # check nan
        if any(np.isnan(val) for val in loss_values.values()):
            self.nan_flag = True
        else:
            self.nan_flag = False

    def test(self, epoch, dataloader):
        # Assuming test on all domains!
        self.eval()
        self.epoch = epoch

        all_loss = np.zeros(self.test_dmn_num)
        count = 0 # for all loss to average
        dataset_len = dataloader.test_datasets[0].__len__() * self.group_len
        # this is for plot
        plt_seq = np.zeros((self.test_dmn_num,dataset_len))

        # test seq save
        l_encode = []
        l_domain = []
        # acc_each = []
        # acc_domain = []
        l_u = torch.zeros(self.num_domain, self.opt.u_dim).to(self.device)
        # l_u_all = torch.zeros(self.num_domain, self.save_sample, self.opt.u_dim).to(self.device)
        l_u_all = []

        for data in dataloader.get_test_data():
            count += 1
            self.__set_input__(data, train=False)

            # set the num2domain dict for the testing dataset
            if count == 1:
                test_num2state = [self.opt.all_domain[idx] for idx in self.domain_seq[:, 0]]

            # test_num = self.seq_len
            # tmp_x_seq = self.x_seq
            # if self.one_hot_seq is not None:
            #     use_one_hot = True
            #     tmp_one_hot_seq = self.one_hot_seq
            #     tmp_domain_seq = self.domain_seq
            # else:
            #     use_one_hot = False

            with torch.no_grad():
                self.x_seq = torch.tensor(self.x_seq).to(self.device).float() # .double()
                # if use_one_hot:
                #     self.one_hot_seq = torch.tensor(tmp_one_hot_seq).double().to(self.device)
                #     self.domain_seq = torch.tensor(tmp_domain_seq).to(self.device)
                
                self.__test_forward__()
                l_encode.append(to_np(self.q_z_seq))
                l_domain.append(self.domain_seq)
                l_u += self.u_seq.sum(1)
                # l_u_all[:, (count-1) * self.factor: count * self.factor, :] = self.u_seq[:, self.save_sample_idx,:]
                l_u_all.append(to_np(self.u_seq))

            np_f_seq = to_np(self.f_seq)
            x_seq = to_np(self.x_seq)
            for i in range(self.test_dmn_num):
                for j in range(self.tmp_batch_size):
                    idx = self.idx_seq[i][j]
                    plt_seq[i, idx * self.group_len + self.seq_len: idx * self.group_len + 2 * self.seq_len] = np_f_seq[i][j]
                    plt_seq[i, idx * self.group_len: idx * self.group_len + self.seq_len] = x_seq[i][j]
            
            loss = (self.y_seq - np_f_seq) ** 2
            # loss shape : Number of test domain x Batch size x test len (Predict Data dim = 1)
            # loss = loss.mean((1,2))
            # balanced loss
            loss = loss.mean((1,2)) * self.tmp_batch_size / self.opt.batch_size
            all_loss += loss

        plt_x = range(dataset_len)
        # this is based on the dataset size now!!
        plt_bar_x = range(9, 144, self.group_len)
        for i in range(self.test_dmn_num):
        
            plt_gt = dataloader.test_datasets[i].data[:dataset_len]
            plt.plot(plt_x, plt_gt, 'b-')
            plt.plot(plt_x, plt_seq[i], 'r-')
            plt.legend(['ground truth', 'prediction'])
            plt.title("{} on epoch {}".format(test_num2state[i] ,self.epoch))
            plt.bar(plt_bar_x, 0.1, width=self.seq_len)

            plt.savefig("{}/{}_{}_epoch.jpg".format(
                self.out_pic_f,
                test_num2state[i], 
                self.epoch), 
                dpi=300, 
                bbox_inches='tight')
            plt.close()

        # test domain loss
        test_loss_mean = all_loss[self.opt.tgt_domain_idx].sum() / (
            self.opt.num_target)

        loss_msg = '[Test][{}] Loss: total average {:.6f}, test loss average {:.6f},  in each domain {}'.format(
            self.epoch, 
            all_loss.mean(),
            test_loss_mean,
            all_loss
        )
        self.__log_write__(loss_msg)
        if self.use_visdom:
            self.__vis_test_error__(test_loss_mean, 'test loss')


        e_all = np.concatenate(l_encode, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        u_all = np.concatenate(l_u_all, axis=1)
        beta_all = to_np(self.beta_seq)
        d_all = dict()
        d_all['domain'] = flat(domain_all)
        d_all['encodeing'] = flat(e_all)
        d_all['u_all'] = flat(u_all)
        # count: how many times u has been accumulated
        d_all['u'] = to_np(l_u / count)
        # d_all['acc_msg'] = acc_msg
        d_all['loss_msg'] = loss_msg
        d_all['beta'] = beta_all
    
        if not self.bayesian_opt and (self.epoch+1) % 100 == 0:
            write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

        if self.epoch == 0:
            write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

        return all_loss.mean(), self.nan_flag

    def my_cat(self, new_u_seq):
        st = new_u_seq[0]
        idx_end = len(new_u_seq)
        for i in range(1, idx_end):
            st = torch.cat((st, new_u_seq[i]), dim=1)
        return st

    def __vis_test_error__(self, loss, title):
        if not self.test_pane_init:
            self.test_pane[title] = self.env.line(X=np.array([self.epoch]),
                                                  Y=np.array([loss]),
                                                  opts=dict(title=title))
            self.test_pane_init = True
        else:
            self.env.line(X=np.array([self.epoch]),
                          Y=np.array([loss]),
                          win=self.test_pane[title],
                          update='append')

    def save(self):
        torch.save(self.netU.state_dict(), self.outf + '/netU_4_dann_compcar.pth')
        torch.save(self.netZ.state_dict(), self.outf + '/netZ.pth')
        torch.save(self.netF.state_dict(), self.outf + '/netF.pth')
        torch.save(self.netR.state_dict(), self.outf + '/netR_4_dann_compcar.pth')
        torch.save(self.netD.state_dict(), self.outf + '/netD.pth')

    def __set_input__(self, data, train=True):
        """
        :param
            x_seq: Number of domain x Batch size x  Data dim
            y_seq: Number of domain x Batch size x Predict Data dim
            (testing: Number of domain x Batch size x test len x Predict Data dim)
            one_hot_seq: Number of domain x Batch size x Number of vertices (domains)
            domain_seq: Number of domain x Batch size x domain dim (1)
            idx_seq: Number of domain x Batch size x 1 (the order in the whole dataset)
            y_value_seq: Number of domain x Batch size x Predict Data dim
        """
        x_seq, y_seq, idx_seq, domain_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data], [d[3][None, :] for d in data]
        
        if train:
            self.x_seq = torch.cat(x_seq, 0).float().to(self.device) # .double()
            self.y_seq = torch.cat(y_seq, 0).float().to(self.device) # .double()
            self.idx_seq = np.concatenate(idx_seq, 0)
            self.domain_seq = torch.cat(domain_seq, 0).to(self.device)
            self.tmp_batch_size = self.x_seq.shape[1]

            # self.prior_mean_tch = torch.from_numpy(self.prior_mean).to(self.device)
            # self.prior_pi_tch = torch.from_numpy(self.prior_pi).to(self.device)
            self.prior_mean_tch = torch.Tensor(self.prior_mean).to(self.device)
            self.prior_pi_tch = torch.Tensor(self.prior_pi).to(self.device)
        else:
            self.x_seq = np.concatenate(x_seq, 0)
            self.y_seq = np.concatenate(y_seq, 0)
            self.idx_seq = np.concatenate(idx_seq, 0)
            self.domain_seq = np.concatenate(domain_seq, 0)
            self.tmp_batch_size = self.x_seq.shape[1]

            self.prior_mean_tch = torch.Tensor(self.prior_mean).to(self.device)
            self.prior_pi_tch = torch.Tensor(self.prior_pi).to(self.device)

    def __train_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        self.u_con_seq = self.netUCon(self.u_seq)

        # if self.opt.d_loss_type == "GRDA_loss":
        if self.use_beta_seq != None:
            self.beta_seq, self.beta_log_var_seq = self.netBeta(
                self.use_beta_seq, self.use_beta_seq)
        else:
            self.tmp_beta_seq = self.generate_beta(self.u_seq)
            self.beta_seq, self.beta_log_var_seq = self.netBeta(
                self.tmp_beta_seq, self.tmp_beta_seq)
        # else:
        #     # CIDA and DANN
        #     self.tmp_beta_seq = self.generate_beta(self.u_seq)
        #     self.beta_seq, self.beta_log_var_seq = self.netBeta(
        #         self.tmp_beta_seq, self.tmp_beta_seq)

        self.beta_U_seq = self.netBeta2U(self.beta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.beta_seq)

        self.r_x_seq = self.netR(self.u_seq)
        self.f_seq = self.netF(self.q_z_seq)

        self.d_seq = self.netD(self.q_z_seq)
        self.loss_D = self.__loss_D__(self.d_seq)

    def __test_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        # if self.opt.d_loss_type == "GRDA_loss":
        if self.use_beta_seq != None:
            self.beta_seq, _ = self.netBeta(self.use_beta_seq,
                                            self.use_beta_seq)
        else:
            self.tmp_beta_seq = self.generate_beta(self.u_seq)
            self.beta_seq, _ = self.netBeta(self.tmp_beta_seq,
                                        self.tmp_beta_seq)
        # else:
        #     # CIDA and DANN
        #     self.tmp_beta_seq = self.generate_beta(self.u_seq)
        #     self.beta_seq, _ = self.netBeta(
        #         self.tmp_beta_seq, self.tmp_beta_seq)
        

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.beta_seq)
        self.f_seq = self.netF(self.q_z_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

        # test
        self.r_x_seq = self.netR(self.u_seq)

    def __optimize__(self):
        loss_value = dict()
        loss_value['D'], loss_value['E_pred'], loss_value['Q_u_x'], loss_value['Q_z_x_u'], loss_value['P_z_x_u'], \
            loss_value['U_concentrate'], loss_value['R'], loss_value['U_beta_R'], loss_value['P_beta_alpha'] \
                = self.__optimize_DUZF__()

        # if self.opt.wgan:
        #     clamp_range = 2.0
        #     for p in self.netD.parameters():
        #         p.data.clamp_(-clamp_range, clamp_range)
        return loss_value

    def contrastive_loss(self, u_con_seq, temperature=1):
        # u_mean = torch.mean(u_con_seq, dim=1, keepdim=True)
        # loss_u_concentrate = ((u_con_seq - u_mean) ** 2).sum(2)
        # loss_u_concentrate = torch.mean(loss_u_concentrate)

        u_con_seq = u_con_seq.reshape(self.tmp_batch_size * self.num_domain,
                                      -1)
        # masks = nn.functional.one_hot(torch.arange(self.tmp_batch_size * self.num_domain)).to(self.device)
        u_con_seq = nn.functional.normalize(u_con_seq, p=2, dim=1)
        logits = torch.matmul(u_con_seq, torch.t(u_con_seq)) / temperature

        # labels
        # base_m = torch.ones(self.tmp_batch_size, self.tmp_batch_size).to(self.device)
        # label = torch.block_diag(*([base_m] * self.num_domain)) / float(self.num_domain) # .softmax(dim=1) #
        # label = torch.arange(self.tmp_batch_size * self.num_domain) + 1
        # label[self.tmp_batch_size * self.num_domain - 1] -= 2
        # label = label.long().to(self.device)

        # We only choose the one that is:
        # 1, belongs to one domain
        # 2, next to each other
        # as the pair that we want to concentrate them, and all the others will be cancel out
        base_m = torch.diag(torch.ones(self.tmp_batch_size - 1),
                            diagonal=1).to(self.device)
        base_m[self.tmp_batch_size - 1, 0] = 1
        base_m = torch.ones(self.tmp_batch_size, self.tmp_batch_size).to(
            self.device) - base_m
        masks = torch.block_diag(*([base_m] * self.num_domain))
        logits = logits - masks * LARGE_NUM

        label = torch.arange(self.tmp_batch_size * self.num_domain).to(
            self.device)
        label = torch.remainder(label + 1, self.tmp_batch_size) + label.div(
            self.tmp_batch_size, rounding_mode='floor') * self.tmp_batch_size

        # print(label)

        loss_u_concentrate = F.cross_entropy(logits, label)
        return loss_u_concentrate

    def __optimize_DUZF__(self):
        self.train()

        self.optimizer_UZF.zero_grad()

        # - E_q[log q(u|x)]
        # u is multi-dimensional
        # not one dimension! be careful1
        loss_q_u_x = torch.mean((0.5 * flat(self.u_log_var_seq)).sum(1), dim=0)

        # - E_q[log q(z|x,u)]
        # remove all the losses about log var and use 1 as var
        loss_q_z_x_u = torch.mean((0.5 * flat(self.q_z_log_var_seq)).sum(1),
                                  dim=0)

        # E_q[log p(z|x,u)]
        # first one is for normal
        loss_p_z_x_u = -0.5 * flat(self.p_z_log_var_seq) - 0.5 * (
            torch.exp(flat(self.q_z_log_var_seq)) +
            (flat(self.q_z_mu_seq) - flat(self.p_z_mu_seq))**2) / flat(
                torch.exp(self.p_z_log_var_seq))
        loss_p_z_x_u = torch.mean(loss_p_z_x_u.sum(1), dim=0)

        # E_q[log p(y|z)]
        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        loss_p_y_z = -F.mse_loss(
            flat(f_seq_source).squeeze(), flat(y_seq_source))

        # use gaussian for u:
        # log_var = flat(self.u_log_var_seq)
        # mu = flat(self.u_mu_seq)
        # loss_u_kld = torch.mean(0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # E_q[log p(u|\beta)]
        # assuming alpha mean = 0
        # loss_beta_alpha = - torch.mean((self.beta_mu_seq ** 2).sum(dim=1))
        var_beta = torch.exp(self.beta_log_var_seq)
        loss_beta_alpha = -torch.mean((var_beta**2).sum(dim=1))

        # use gaussian for beta
        # log_var = flat(self.beta_log_var_seq)
        # mu = flat(self.beta_mu_seq)
        # loss_beta_kld = torch.mean(0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # loss for u and beta
        beta_t = self.beta_U_seq.unsqueeze(dim=1).expand(
            -1, self.tmp_batch_size, -1)
        # now beta_t is domain x batch x domain idx dim
        loss_p_u_beta = ((self.u_seq - beta_t)**2).sum(2)
        loss_p_u_beta = -torch.mean(loss_p_u_beta)

        # # concentrate loss
        # self.u_mean = torch.mean(self.u_seq, dim=1, keepdim=True)
        # loss_u_concentrate = ((self.u_seq - self.u_mean) ** 2).sum(2)
        # loss_u_concentrate = torch.mean(loss_u_concentrate)
        # change to contrastive learning
        loss_u_concentrate = self.contrastive_loss(self.u_con_seq)

        # reconstruction loss
        loss_p_x_u = ((flat(self.x_seq) - flat(self.r_x_seq))**2).sum(1)
        loss_p_x_u = -torch.mean(loss_p_x_u)

        # E gan loss
        if self.opt.lambda_gan != 0:
            loss_E_gan = -self.loss_D
        else:
            loss_E_gan = torch.tensor(0,
                                      dtype=torch.double,
                                      device=self.opt.device)

        # # shape problem
        # loss_E = self.opt.lambda_reconstruct * loss_R + loss_E_gan * self.opt.lambda_gan - (- loss_p_y_z + loss_q_u_x + loss_u_kld - self.opt.lambda_u_concentrate * loss_u_concentrate + loss_q_z_x_u + loss_p_z_x_u)
        loss_E = loss_E_gan * self.opt.lambda_gan + self.opt.lambda_u_concentrate * loss_u_concentrate - (
            self.opt.lambda_reconstruct * loss_p_x_u + self.opt.lambda_beta *
            loss_p_u_beta + self.opt.lambda_beta_alpha * loss_beta_alpha +
            loss_p_y_z + loss_q_u_x + loss_q_z_x_u + loss_p_z_x_u)

        self.optimizer_D.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_UZF.zero_grad()
        loss_E.backward()

        self.optimizer_D.step()
        self.optimizer_UZF.step()

        return self.loss_D.item(), -loss_p_y_z.item(), loss_q_u_x.item(
        ), loss_q_z_x_u.item(), loss_p_z_x_u.item(), loss_u_concentrate.item(
        ), -loss_p_x_u.item(), -loss_p_u_beta.item(), -loss_beta_alpha.item()

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n")

    def __vis_loss__(self, loss_values):
        if self.epoch == 0:
            self.panes = {
                loss_name: self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title='loss for {} on epochs'.format(loss_name)))
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(X=np.array([self.epoch]),
                              Y=np.array([loss_values[loss_name]]),
                              win=self.panes[loss_name],
                              update='append')

    def __init_weight__(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # print("init linear weight!")
                nn.init.normal_(m.weight, mean=0, std=0.01)
                #                 nn.init.normal_(m.weight, mean=0, std=0.1)
                #                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)


class IDI(BaseModel):
    """
    IDI (Inference Domain Index) Model
    """

    def __init__(self, opt, search_space=None):
        super(IDI, self).__init__(opt)

        self.bayesian_opt = False
        if search_space != None:
            self.bayesian_opt = True

        # self.netU = GMMNet(opt).to(opt.device)
        self.netU = UNet(opt).to(opt.device)
        self.netUCon = UConcenNet(opt).to(opt.device)
        # self.netZ = Q_ZNet(opt).to(opt.device)
        self.netZ = Q_ZNet_beta(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netR = ReconstructNet(opt).to(opt.device)

        self.netBeta = BetaNet(opt).to(opt.device).float()
        self.netBeta2U = Beta2UNet(opt).to(opt.device).float()
        # TODO:实现Beta2UNet，module里相关替换

        # for DANN
        if self.opt.d_loss_type == "DANN_loss":
            self.netD = ClassDiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_dann__
            # self.generate_beta = self.__u_mean__
            self.generate_beta = self.__reconstruct_u_graph__
        # this is for CIDA
        elif self.opt.d_loss_type == "CIDA_loss":
            self.netD = DiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_cida__
            self.generate_beta = self.__u_mean__
        # for grda
        elif self.opt.d_loss_type == "GRDA_loss":
            self.netD = GraphDNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_grda__
            self.generate_beta = self.__reconstruct_u_graph__
        self.__init_weight__()

        if self.opt.use_pretrain_R:
            pretrain_model_U = torch.load(self.opt.pretrain_U_path)
            pretrain_model_R = torch.load(self.opt.pretrain_R_path)

            self.netU.load_state_dict(pretrain_model_U)
            self.netR.load_state_dict(pretrain_model_R)

        if self.opt.fix_u_r:
            UZF_parameters = list(self.netZ.parameters()) + list(
                self.netF.parameters())
        else:
            UZF_parameters = list(self.netU.parameters()) + list(
                self.netZ.parameters()) + list(self.netF.parameters()) + list(
                    self.netR.parameters()) + list(self.netUCon.parameters())

        # if self.opt.d_loss_type == "GRDA_loss":
        UZF_parameters += list(self.netBeta.parameters()) + list(
            self.netBeta2U.parameters())

        self.optimizer_UZF = optim.Adam(UZF_parameters,
                                        lr=opt.init_lr,
                                        betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(),
                                      lr=opt.init_lr,
                                      betas=(opt.beta1, 0.999))
        self.lr_scheduler_UZF = TransformerLRScheduler(
            optimizer=self.optimizer_UZF,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_e,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.num_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)
        self.lr_scheduler_D = TransformerLRScheduler(
            optimizer=self.optimizer_D,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_d,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.num_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)

        self.lr_schedulers = [self.lr_scheduler_UZF, self.lr_scheduler_D]
        self.loss_names = [
            'D', 'E_pred', 'Q_u_x', 'Q_z_x_u', 'P_z_x_u', 'U_beta_R',
            'U_concentrate', 'R', 'P_beta_alpha'
        ]

        # for mds(u)
        self.embedding = MDS(n_components=self.opt.beta_dim,
                             dissimilarity='precomputed')

    def __u_mean__(self, u_seq):
        return u_seq.mean(1).detach()  # , u_seq.mean(1), np.exp(0.5)

    def __reconstruct_u_graph__(self, u_seq):
        with torch.no_grad():
            A = torch.zeros(self.num_domain, self.num_domain)
            new_u = u_seq.detach()
            # ~ Wasserstein Loss 
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            for i in range(self.num_domain):
                for j in range(i + 1, self.num_domain):
                    A[i][j] = loss(new_u[i], new_u[j])
                    A[j][i] = A[i][j]
            
            A_np = to_np(A)
            # bound = np.sort(A_np.flatten())[int(self.num_domain**2 * 1 / 3)]
            bound = np.sort(A.flatten())[int(self.num_domain ** 2 * 1 / 4)]
            # A_dis = A
            # generate self.A
            self.A = (A_np < bound)

            # calculate the beta seq
            mu_beta = self.embedding.fit_transform(A_np)
            mu_beta = torch.from_numpy(mu_beta).to(self.device)
            # mu_beta = F.normalize(mu_beta)

            # new normalization:
            # ((a - a.mean(0, keepdim=True)) / (a.std(0, keepdim=True) + 1e-12))
            mu_beta_mean = mu_beta.mean(0, keepdim=True)
            mu_beta_std = mu_beta.std(0, keepdim=True)
            mu_beta_std = torch.maximum(mu_beta_std, torch.ones_like(mu_beta_std) * 1e-12)
            mu_beta = (mu_beta - mu_beta_mean) / mu_beta_std
            
            
            # std = torch.ones_like(mu_beta).to(self.device) * self.opt.sigma_beta
            # eps = torch.randn_like(std).to(self.device)
            # beta = mu_beta + std * eps

            return mu_beta  # , A

    # def __reconstruct_u_graph__(self, u_seq):
    #     A = np.zeros((self.opt.num_domain, self.opt.num_domain))

    #     new_u = to_np(u_seq)
    #     A = np.zeros((self.num_domain, self.num_domain))
    #     tmp_size = new_u.shape[1]
    #     for i in range(self.num_domain):
    #         for j in range(i + 1, self.num_domain):
    #             M = ot.dist(new_u[i], new_u[j])
    #             a = np.ones(tmp_size) / tmp_size
    #             b = np.ones(tmp_size) / tmp_size
    #             Wd = ot.emd2(a, b, M)
    #             A[i][j] = Wd
    #             A[j][i] = Wd
    #     bound = np.sort(A.flatten())[int(self.num_domain**2 * 1 / 3)]
    #     # bound = np.sort(A.flatten())[int(self.num_domain ** 2 * 1 / 4)]
    #     A_dis = A
    #     # generate self.A
    #     self.A = (A < bound)

    #     # calculate the beta seq
    #     mu_beta = self.embedding.fit_transform(A_dis)
    #     mu_beta = torch.from_numpy(mu_beta).to(self.device)
    #     mu_beta = F.normalize(mu_beta)
    #     # std = torch.ones_like(mu_beta).to(self.device) * self.opt.sigma_beta
    #     # eps = torch.randn_like(std).to(self.device)
    #     # beta = mu_beta + std * eps

    #     return mu_beta  # , A
    #     # return beta

    def __loss_D_dann__(self, d_seq):
        # this is for DANN
        return F.nll_loss(flat(d_seq),
                          flat(self.domain_seq))  # , self.u_seq.mean(1)

    def __loss_D_cida__(self, d_seq):
        # use L1 instead of L2
        # this is for CIDA
        return F.l1_loss(flat(d_seq),
                         flat(self.u_seq.detach()))  # , self.u_seq.mean(1)

    # grda loss
    def __loss_D_grda__(self, d_seq):
        # first build the graph
        # already generate in \beta_seq
        A = self.A

        criterion = nn.BCEWithLogitsLoss()
        d = d_seq
        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio
        # A is the adjancency matrix
        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v, A=A)

        errorD_connected = torch.zeros((1, )).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1, )).to(self.device)  # .double()

        count_connected = 0
        count_disconnected = 0

        for i in range(self.opt.sample_v):
            v_i = sub_graph[i]
            # no self loop version!!
            for j in range(i + 1, self.opt.sample_v):
                v_j = sub_graph[j]
                label = torch.full(
                    (self.tmp_batch_size, ),
                    A[v_i][v_j],
                    device=self.device,
                )
                # dot product
                if v_i == v_j:
                    idx = torch.randperm(self.tmp_batch_size)
                    output = (d[v_i][idx] * d[v_j]).sum(1)
                else:
                    output = (d[v_i] * d[v_j]).sum(1)

                if A[v_i][v_j]:  # connected
                    errorD_connected += criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += criterion(output, label)
                    count_disconnected += 1

        # prevent nan
        if count_connected == 0:
            count_connected = 1
        if count_disconnected == 0:
            count_disconnected = 1

        errorD = 0.5 * (errorD_connected / count_connected +
                        errorD_disconnected / count_disconnected)
        # this is a loss balance
        return errorD * self.num_domain  #, beta
        # return errorD

    def __sub_graph__(self, my_sample_v, A):
        if np.random.randint(0, 2) == 0:
            return np.random.choice(self.num_domain,
                                    size=my_sample_v,
                                    replace=False)

        # subsample a chain (or multiple chains in graph)
        left_nodes = my_sample_v
        choosen_node = []
        vis = np.zeros(self.num_domain)
        while left_nodes > 0:
            chain_node, node_num = self.__rand_walk__(vis, left_nodes, A)
            choosen_node.extend(chain_node)
            left_nodes -= node_num

        return choosen_node

    # for graph random sampling:
    def __rand_walk__(self, vis, left_nodes, A):
        chain_node = []
        node_num = 0
        # choose node
        node_index = np.where(vis == 0)[0]
        st = np.random.choice(node_index)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1

        cur_node = st
        while left_nodes > 0:
            nx_node = -1

            node_to_choose = np.where(vis == 0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(node_to_choose,
                                              num,
                                              replace=False)

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and doesn't visit
                    if A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break
        return chain_node, node_num
