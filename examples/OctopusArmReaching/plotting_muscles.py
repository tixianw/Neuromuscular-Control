import sys

sys.path.append("../../")
# from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# from matplotlib.collections import LineCollection
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import gridspec
import numpy as np

# from numpy.linalg import norm
# from scipy.signal import butter, filtfilt
np.seterr(divide="ignore", invalid="ignore")
# from neuromuscular_control.utils import _diff, _aver, _diff_kernel, _aver_kernel
# import matplotlib.colors as mcolors
# import matplotlib as mpl
# from matplotlib.ticker import (MultipleLocator,
#                                FormatStrFormatter,
#                                AutoMinorLocator)
# from neuromuscular_control.utils import gaussian
import os


def isninf(a):
    return np.all(np.isfinite(a))


choice = 0
folder_name = "Data/"
if choice == 0:  ### equilibrium, initialization, backstepping,
    file_name = "test"
else:
    pass

data = np.load(folder_name + file_name + ".npy", allow_pickle="TRUE").item()

n_elem = data["model"]["arm"]["n_elem"]
L = data["model"]["arm"]["L"]
radius = data["model"]["arm"]["radius"]
if radius[0] == radius[1]:
    r = ((2 * radius) / (2 * radius[0])) ** 2 * 50
else:
    r = ((2 * radius) / (2 * radius[0])) ** 2 * 100
E = data["model"]["arm"]["E"]
final_time = data["model"]["numerics"]["final_time"]
dt = data["model"]["numerics"]["step_size"]
t = data["t"]
s = data["model"]["arm"]["s"]
s_mean = (s[1:] + s[:-1]) / 2
ds = s[1] - s[0]
arm = data["arm"]
muscle = data["muscle"]
neuron = data["neuron"]
sensor = data["sensor"]
s_bar_idx = sensor[-1]["s_bar"]
save_step_skip = data["model"]["numerics"]["step_skip"]
flags = data["model"]["flags"]
flag_target = flags[1]
flag_obs = flags[2]
if flag_target:
    target = data["model"]["target"]
if flag_obs:
    Obs = data["model"]["obstacle"]
    N_obs = Obs["N_obs"]
    print(N_obs, "obstacles")
    pos_obs = Obs["pos_obs"]
    r_obs = Obs["r_obs"]
    len_obs = Obs["len_obs"]
position = arm[-1]["position"][:, :2, :]
orientation = arm[-1]["orientation"][:, 1:, :-1, :]
# velocity = arm[-1]['velocity']
# vel_mag = np.sqrt(np.einsum('ijn,ijn->in', velocity, velocity))


def plot_arm():
    min_var_x = -0.8 * L
    max_var_x = 1.1 * L
    min_var_y = -0.8 * L
    max_var_y = 1.1 * L
    fig = plt.figure(figsize=(10 * 0.6, 10 * 0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if flag_obs:
        alpha0 = 0.8
        name_obstacle = locals()
        for o in range(N_obs):
            name_obstacle["obstacle" + str(o)] = plt.Circle(
                (pos_obs[o, 0], pos_obs[o, 1]), r_obs[o], color="grey", alpha=alpha0
            )
            ax0.add_artist(name_obstacle["obstacle" + str(o)])
    i = -1
    ax0.scatter(
        position[i, 0, :], position[i, 1, :], s=r, marker="o", alpha=1, zorder=2
    )
    # ax0.scatter(position[i,0,idx],position[i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
    ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
    ax0.axis("off")


plot = 1
flag_savefig = 0
if plot:
    plot_arm()
    plt.show()

if choice == 0:
    video = 2  # 1 #
else:
    pass
save_flag = 0

## No target, only arm
if video == 0:
    max_var = L * 1.1
    min_var = -L / 2
    idx = -1
    min_var_x = min(
        np.amin(position[0, 0, :]) * 1.1,
        np.amin(position[-1, 0, :]) * 1.1,
        min_var * 1.01,
    )
    max_var_x = max(
        np.amax(position[-1, 0, :]) * 1.1,
        np.amax(position[0, 0, :]) * 1.1,
        max_var * 1.01,
    )
    min_var_y = min(np.amin(position[-1, 1, :]) * 1.1, min_var * 1.01)
    max_var_y = max_var * 1.01
    # dist = neuron[idx]['dist'][:,:]
    fig = plt.figure(figsize=(10 * 0.6, 10 * 0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if save_flag:
        factor1 = 5  # min(int(1000 / save_step_skip), 1) # 5
        if choice == 0:
            name = file_name
    else:
        factor1 = int(2000 / save_step_skip)
        name = "trash"
    fps = 5  # 10
    os.mkdir("Videos/")
    video_name = "Videos/" + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        for jj in range(int((len(t) - 1) / factor1) + 1):  # +1
            i = jj * factor1
            time = i / (len(t) - 1) * final_time
            # idx = np.argmin(dist[i, :])
            ax0.cla()
            if flag_obs:
                alpha0 = 0.8
                name_obstacle = locals()
                for o in range(N_obs):
                    name_obstacle["obstacle" + str(o)] = plt.Circle(
                        (pos_obs[o, 0], pos_obs[o, 1]),
                        r_obs[o],
                        color="grey",
                        alpha=alpha0,
                    )
                    ax0.add_artist(name_obstacle["obstacle" + str(o)])
            ax0.scatter(
                position[i, 0, :], position[i, 1, :], s=r, marker="o", alpha=1, zorder=2
            )
            # ax0.scatter(position[i,0,idx],position[i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
            ax0.text(L * 0.1, max_var * 1.05, "t: %.3f s" % (time), fontsize=12)
            angle = np.linspace(0, 2 * np.pi, 100)
            # distance = norm(target[0,:])
            # ax0.plot(target[0,0]+distance*np.cos(angle), target[0,1]+distance*np.sin(angle), ls='--', color='black')
            ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
            if not save_flag:
                plt.pause(0.001)
            else:
                writer.grab_frame()
            # break
            if not isninf(position):
                break

##
elif video == 1:
    max_var = max(
        L * 1.1, np.amax(target[:, 0]) * 1.1
    )  # max(L*1.1, target[0]*1.1) # 1.5
    min_var = min(-L / 2, np.amin(target[:, 0]) * 1.1)  # /3 # /4
    idx = -1
    min_var_x = min(
        np.amin(position[0, 0, :]) * 1.1,
        np.amin(position[-1, 0, :]) * 1.1,
        min_var * 1.01,
    )
    max_var_x = max(
        np.amax(position[-1, 0, :]) * 1.1,
        np.amax(position[0, 0, :]) * 1.1,
        max_var * 1.01,
    )
    min_var_y = min(np.amin(position[-1, 1, :]) * 1.1, min_var * 1.01)
    max_var_y = max_var * 1.01
    dist = neuron[idx]["dist"][:, :]
    fig = plt.figure(figsize=(10 * 0.6, 10 * 0.6))
    ax0 = fig.add_subplot(1, 1, 1)
    if save_flag:
        factor1 = 5  # min(int(1000 / save_step_skip), 1) # 5
        if choice == 0:
            name = file_name  # + '_bend_vel'
    else:
        factor1 = int(2000 / save_step_skip)
        name = "trash"
    fps = 5  # 10
    os.mkdir("Videos/")
    video_name = "Videos/" + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        for jj in range(int((len(t) - 1) / factor1) + 1):  # +1
            i = jj * factor1
            time = i / (len(t) - 1) * final_time
            idx = np.argmin(dist[i, :])
            ax0.cla()
            if flag_obs:
                alpha0 = 0.8
                name_obstacle = locals()
                for o in range(N_obs):
                    name_obstacle["obstacle" + str(o)] = plt.Circle(
                        (pos_obs[o, 0], pos_obs[o, 1]),
                        r_obs[o],
                        color="grey",
                        alpha=alpha0,
                    )
                    ax0.add_artist(name_obstacle["obstacle" + str(o)])
            ax0.scatter(
                position[i, 0, :], position[i, 1, :], s=r, marker="o", alpha=1, zorder=2
            )
            ax0.scatter(
                position[i, 0, idx],
                position[i, 1, idx],
                s=r[idx],
                marker="o",
                color="red",
                alpha=1,
                zorder=3,
            )
            ax0.text(L * 0.1, max_var * 1.05, "t: %.3f s" % (time), fontsize=12)
            ax0.scatter(
                target[i, 0],
                target[i, 1],
                s=200,
                marker="*",
                label="target point",
                zorder=1,
            )
            angle = np.linspace(0, 2 * np.pi, 100)
            # distance = norm(target[0,:])
            # ax0.plot(target[0,0]+distance*np.cos(angle), target[0,1]+distance*np.sin(angle), ls='--', color='black')
            ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
            if not save_flag:
                plt.pause(0.001)
            else:
                writer.grab_frame()
            # break
            if not isninf(position):
                break


## V and I from cable equation
elif video == 2:
    max_var = L * 1.1
    min_var = -L / 2
    idx = -1
    min_var_x = (
        -0.8 * L
    )  # min(np.amin(position[0,0,:])*1.1, np.amin(position[-1,0,:])*1.1, min_var*1.01)
    max_var_x = (
        1.1 * L
    )  # max(np.amax(position[-1,0,:])*1.1, np.amax(position[0,0,:])*1.1, max_var*1.01)
    min_var_y = -0.8 * L  # min(np.amin(position[-1,1,:])*1.1, min_var*1.01)
    max_var_y = 1.1 * L  # max_var*1.01
    var1 = neuron[idx]["I"][:, :, :]
    var2 = neuron[idx]["V"][:, :, :]
    var3 = muscle[idx]["u"][:, :, :]
    kappa = arm[idx]["kappa"][:, :]
    min_var1 = []
    max_var1 = []
    min_var2 = []
    max_var2 = []
    min_var3 = []
    max_var3 = []
    for i in range(var1.shape[1]):
        min_var1.append(np.amin(var1[:, i, :]))
        max_var1.append(np.amax(var1[:, i, :]))
        min_var2.append(np.amin(var2[:, i, :]))
        max_var2.append(np.amax(var2[:, i, :]))
        min_var3.append(np.amin(var3[:, i, :]))
        max_var3.append(np.amax(var3[:, i, :]))
    var1_min = -1  # -500 # min(min_var1)*1-max(max_var1)*0.1
    var1_max = 1000  # 500 # max(max_var1)*1.1
    var2_min = -20  # min(min_var2)-1 #
    var2_max = 60  # max(max_var2)+1 #
    var3_min = -0.05  # min(min_var3)*1-max(max_var3)*0.1
    var3_max = 1.05  # max(max_var3)*1.1
    min_kappa = -100  # np.amin(kappa) #
    max_kappa = 150  # np.amax(kappa) #
    fig = plt.figure(figsize=(30 * 0.6, 9 * 0.6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(2, 3, 2)
    ax2 = fig.add_subplot(2, 3, 3)
    ax3 = fig.add_subplot(2, 3, 5)
    ax4 = fig.add_subplot(2, 3, 6)
    if save_flag:
        factor1 = 1  # 5
        name = file_name  # + '_shooting' # 'sensor_shooting' # '_zero_current' # '_backstepping_constant_u' #
    else:
        factor1 = 5  # int(5000 / save_step_skip) # 50 #
        name = "trash"
    slow_factor = 2  # 0.5 #
    fps = 100 / slow_factor
    os.mkdir("Videos/")
    video_name = "Videos/" + name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        var1 = neuron[idx]["I"][:, :, :]
        var2 = neuron[idx]["V"][:, :, :]
        var3 = muscle[idx]["u"][:, :, :]
        kappa = arm[idx]["kappa"][:, :]
        for jj in range(int((len(t) - 1) / factor1) + 1):  # +1
            i = jj * factor1
            time = i / (len(t) - 1) * final_time
            ax0.cla()
            if flag_obs:
                alpha0 = 0.8
                name_obstacle = locals()
                for o in range(N_obs):
                    name_obstacle["obstacle" + str(o)] = plt.Circle(
                        (pos_obs[o, 0], pos_obs[o, 1]),
                        r_obs[o],
                        color="grey",
                        alpha=alpha0,
                    )
                    ax0.add_artist(name_obstacle["obstacle" + str(o)])
            ax0.scatter(
                position[i, 0, :], position[i, 1, :], s=r, marker="o", alpha=1, zorder=2
            )
            # ax0.scatter(position[i,0,idx],position[i,1,idx], s=r[idx], marker='o', color='red', alpha=1,zorder=3)
            ax0.text(L * 0.1, max_var * 1.05, "t: %.3f s" % (time), fontsize=12)
            if flag_target:
                ax0.scatter(
                    target[i, 0],
                    target[i, 1],
                    s=100,
                    marker="*",
                    label="target point",
                    zorder=1,
                )
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            for ii in range(var1.shape[1]):
                ax1.plot(s, var1[i, ii, :])
                ax2.plot(s, var2[i, ii, :])
                ax3.plot(s, var3[i, ii, :])
                # ax3.plot(s, desired_u[ii, :], ls='--', color='k')
            ax4.plot(s[1:-1], kappa[i, :])
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax1.set_ylabel("$I_i$", fontsize=12)
            ax2.set_ylabel("$V_i$", fontsize=12)
            ax3.set_ylabel("$u_i$", fontsize=12)
            ax3.set_xlabel("$s$", fontsize=12)
            ax4.set_xlabel("$s$", fontsize=12)
            ax4.set_ylabel("$\\kappa$", fontsize=12)
            ax0.axis([min_var_x, max_var_x, min_var_y, max_var_y])
            ax1.axis([-0.01, L + 0.01, var1_min, var1_max])
            ax2.axis([-0.01, L + 0.01, var2_min, var2_max])
            ax3.axis([-0.01, L + 0.01, var3_min, var3_max])
            ax4.axis([-0.01, L + 0.01, min_kappa, max_kappa])
            # break
            if not save_flag:
                plt.pause(0.001)
            else:
                writer.grab_frame()
            if not isninf(position):
                break

plt.show()
