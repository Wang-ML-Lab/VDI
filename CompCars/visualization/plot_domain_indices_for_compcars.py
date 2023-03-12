import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import pickle
import matplotlib
import sys

np.random.seed(0)
random.seed(0)


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


# plt.rcParams['text.usetex'] = True


def draw_2_div(data, num_domain):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
    cmap = matplotlib.cm.get_cmap('rainbow')
    l_color = [cmap(ele)[:3] for ele in np.linspace(0, 1, num_domain)]

    title_size = 20
    aspect_ratio = 0.6

    # first plot, Colors Indicate Viewpoints
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
    cmap = matplotlib.cm.get_cmap('hsv')
    l_color = [cmap(ele)[:3] for ele in np.linspace(0.05, 0.95, 5)]
    VIEW = [
        'Front (F)', 'Front-side (FS)', 'Side (S)', 'Rear-side (RS)',
        'Rear (R)'
    ]

    for i in range(num_domain):
        draw_label = i
        print("domain: {}".format(draw_label))
        x = data[draw_label, 0]
        y = data[draw_label, 1]
        if i < 5:
            ax.scatter(x, y, color=l_color[i % len(l_color)], label=VIEW[i])
        else:
            ax.scatter(x, y, color=l_color[i % len(l_color)])
        ax.annotate(str(draw_label), (x, y), xytext=(x, y), weight='bold')
    # ax.legend(fontsize=12)
    ax.legend(fontsize=12, bbox_to_anchor=(1.04, 0.5), loc="center left")

    ax.set_title("{} (Colors Indicate Viewpoints)".format(r"$\beta$"),
                 y=1.0,
                 pad=11)
    ax.title.set_size(title_size)
    ax.set_aspect(aspect_ratio)
    plt.savefig("{}/{}_colors_indicate_viewpoints.pdf".format(
        save_folder, "Beta"),
                bbox_inches='tight',
                dpi=300,
                format='pdf')
    # plt.show()
    plt.clf()

    # second plot, Colors Indicate YOMs
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
    cmap = matplotlib.cm.get_cmap('rainbow')
    l_color = [cmap(ele)[:3] for ele in np.linspace(0, 1, 6)]

    YOM = 2009
    for i in range(num_domain):
        draw_label = i
        print("domain: {}".format(draw_label))
        x = data[draw_label, 0]
        y = data[draw_label, 1]

        color = l_color[i // 5]
        if i % 5 == 0:
            ax.scatter(x, y, color=color, label=YOM + i // 5)
        else:
            ax.scatter(x, y, color=color)
        ax.annotate(str(draw_label), (x, y), xytext=(x, y), weight='bold')
    # ax.legend(fontsize=12)
    ax.legend(fontsize=12, bbox_to_anchor=(1.04, 0.5), loc="center left")
    ax.set_title("{} (Colors Indicate YOMs)".format(r"$\beta$"), y=1.0, pad=11)
    ax.title.set_size(title_size)
    ax.set_aspect(aspect_ratio)
    plt.savefig("{}/{}_colors_indicate_YOMs.pdf".format(save_folder, "Beta"),
                bbox_inches='tight',
                dpi=300,
                format='pdf')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    result = sys.argv[1]
    save_folder = sys.argv[2]
    info = read_pickle(result)

    print(info['acc_msg'])
    data_domain = info['domain']

    num_domain = int(np.max(data_domain)) + 1

    beta = info['beta']
    draw_2_div(beta, num_domain)
