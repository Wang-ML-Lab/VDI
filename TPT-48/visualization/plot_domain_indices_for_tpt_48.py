import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import pickle
import matplotlib
import pandas as pd
import sys

np.random.seed(0)
random.seed(0)


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def draw_2_div(data, num_domain):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
    cmap = matplotlib.cm.get_cmap('rainbow')
    title_size = 20
    aspect_ratio = 0.75
    long_latitude = pd.read_csv(
        "visualization/usa_map/longitude-latitude-normalize.csv")

    # first plot, Colors Indicate Latitude
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
    cmap = matplotlib.cm.get_cmap('rainbow')

    for i in range(num_domain):
        draw_label = all_domain[i]
        print("domain: {}".format(draw_label))
        x = data[i, 0]
        y = data[i, 1]
        my_latitude = long_latitude.loc[long_latitude['state'] ==
                                        draw_label]['latitude']
        ax.scatter(x, y, color=cmap(my_latitude)[:3])
        ax.annotate(str(draw_label), (x, y), xytext=(x, y), weight='bold')
    ax.set_title("{} (Colors Indicate Latitude)".format(r"$\beta$"),
                 y=1.0,
                 pad=11)
    ax.title.set_size(title_size)
    ax.set_aspect(aspect_ratio)
    plt.savefig("{}/{}_colors_indicate_latitude.pdf".format(
        save_folder, "Beta"),
                dpi=300,
                format='pdf',
                bbox_inches='tight')  #
    # plt.show()
    plt.clf()

    # second plot, Colors Indicate Longitude
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))

    for i in range(num_domain):
        draw_label = all_domain[i]
        print("domain: {}".format(draw_label))
        x = data[i, 0]
        y = data[i, 1]
        my_longitude = long_latitude.loc[long_latitude['state'] ==
                                         draw_label]['longitude']
        ax.scatter(x, y, color=cmap(my_longitude)[:3])
        ax.annotate(str(draw_label), (x, y), xytext=(x, y), weight='bold')
    ax.set_title("{} (Colors Indicate Longitude)".format(r"$\beta$"),
                 y=1.0,
                 pad=11)
    ax.title.set_size(title_size)
    ax.set_aspect(aspect_ratio)
    plt.savefig("{}/{}_colors_indicate_longitude.pdf".format(
        save_folder, "Beta"),
                bbox_inches='tight',
                dpi=300,
                format='pdf')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    result = sys.argv[1]
    save_folder = sys.argv[2]
    config_pth = sys.argv[3]
    info = read_pickle(result)

    print(info['loss_msg'])
    data_domain = info['domain']

    import json
    with open(config_pth) as json_config:
        config = json.load(json_config)

    src_domain = config['src_domain']
    tgt_domain = config['tgt_domain']
    all_domain = config['all_domain']

    num_domain = int(np.max(data_domain)) + 1

    beta = info['beta']
    draw_2_div(beta, num_domain)
