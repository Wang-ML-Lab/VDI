# draw the local domain index
result_folder = opt.outf
result_fd_cmd = opt.outf_vis

result_save_vis_pth = result_folder + "/visualization"
if not os.path.exists(result_save_vis_pth):
    os.mkdir(result_save_vis_pth)

concate_pic_pth = result_folder + "/concate_pic"
if not os.path.exists(concate_pic_pth):
    os.mkdir(concate_pic_pth)

result_cmd = result_fd_cmd + "/{}_pred.pkl".format(opt.num_epoch - 1)
result_save_vis_cmd = result_fd_cmd + "/visualization"
concate_pic_cmd = result_fd_cmd + "/concate_pic"
result_config_cmd = result_fd_cmd + "/config.json"

os.system("python local_draw/plot_u_tsne_2_div.py {} {} {}".format(result_cmd, result_save_vis_cmd, result_config_cmd))