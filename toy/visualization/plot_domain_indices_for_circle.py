import os
import scipy
from easydict import EasyDict
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import matplotlib
import sys
import pickle

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    result = sys.argv[1]
    save_folder = sys.argv[2]
    info = read_pickle(result)
    plot_data = 'beta'

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
    print(info['data'].shape)
    print(info['acc_msg'])

    data_raw = info[plot_data]
    data_domain = info['domain']

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    # pca = TSNE(n_components=1)
    pca = PCA(n_components=1)
    data_all = pca.fit_transform(data_raw)

    label_all = data_domain

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))

    cmap = matplotlib.cm.get_cmap('rainbow')
    num = int(np.max(label_all)) + 1
    l_color = [cmap(ele)[:3] for ele in np.linspace(0, 1, num)]

    data_all = (data_all - min(data_all)) / (max(data_all) - min(data_all))
    x_all = np.linspace(0, 1, 30)

    # print(data_all)
    # print(x_all)
    corr = np.corrcoef(x_all.T, data_all[::,0])
    print('corr: ', corr)
    corr_num = scipy.stats.pearsonr(x_all.T, data_all[::,0])
    print('corr: ', corr_num)

    fsize = 18
    fsize_legend = 16
    fsize_tick = 16

    plt.ylabel('Normalized Estimated Domain Indices', fontsize = fsize)
    plt.xlabel('Normalized True Domain Indices', fontsize = fsize)
    plt.yticks(fontsize = fsize_tick)
    plt.xticks(fontsize = fsize_tick)
    props = dict(boxstyle='Square', facecolor='white')
    ax.text(0.01, 0.95, 'Correlation={:.2f}'.format(corr_num[0]), fontsize=fsize_legend, bbox=props)

    plt.plot(x_all, data_all[::], 'ko')
    plt.savefig("{}/c30_beta_new_tsne.pdf".format(save_folder),format = 'pdf', bbox_inches='tight',dpi=300, pad_inches = 0)
