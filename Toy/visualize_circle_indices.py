import os

dates = "2023-03-10"  # filling your own dates for experiments
time = ["14", "14", "04"
        ]  # filling the time for experiments. format: hour, miniute, second
result_folder = r"result_save/{} {}:{}:{}".format(dates, time[0], time[1],
                                                  time[2])
result_fd_cmd = r"result_save/{}\ {}\:{}\:{}".format(dates, time[0], time[1],
                                                     time[2])

result_save_vis_pth = result_folder + "/visualization"
if not os.path.exists(result_save_vis_pth):
    os.mkdir(result_save_vis_pth)

result_cmd = result_fd_cmd + "/799_pred.pkl"
result_save_vis_cmd = result_fd_cmd + "/visualization"

os.system(
    "python visualization/plot_domain_indices_for_circle.py {} {}".format(
        result_cmd, result_save_vis_cmd))
