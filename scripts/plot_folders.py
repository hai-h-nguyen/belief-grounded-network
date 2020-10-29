import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import argparse
from plot_utils import *
import os

def getLabel(curr_dir):
    if curr_dir == 'ahch':
        return 'Ah-Ch'

    if curr_dir == 'bgn':
        return 'Ah-Ch + BGN'

    if curr_dir == 'ahcs':
        return 'Ah-Cs'

    if curr_dir == 'ahcb':
        return 'Ah-Cb'        

    if curr_dir == 'abcb':
        return 'Ab-Cb'

# colors = sns.color_palette("bright", 10)
mpl.rcParams['pdf.fonttype'] = 42
sns.set_style('darkgrid')
cmap = plt.get_cmap('tab10')

parser = argparse.ArgumentParser()
parser.add_argument('--folder', nargs='+', type=str)
parser.add_argument('--sarsop', nargs='+', default=None, type=float)
parser.add_argument('--random', nargs='+', default=None, type=float)
parser.add_argument('--window', nargs='+', default=10, type=int)
parser.add_argument('--labels', nargs='+', default=None, type=str)
parser.add_argument('--mode', nargs='+', default='training', type=str)
parser.add_argument('--domain', nargs='+', type=str)
parser.add_argument("--text", type=int, default=14)

args = parser.parse_args()

mpl.rc('xtick', labelsize=args.text) 
mpl.rc('ytick', labelsize=args.text)

color_idx = 0

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

plot_indices = ((0, 0), (0, 1), (1, 0), (1, 1))

for i in range(len(args.folder)):
    # Go to the specified folder
    folder = args.folder[i]
    color_idx = 0
    x = plot_indices[i][0]
    y = plot_indices[i][1]
    for root, dirs, _ in os.walk(folder):
        for d in dirs:
            current_dir = os.path.join(root,d)
            label = getLabel(d)
            # Go into this directory to search for files
            files = os.listdir(current_dir)

            if (len(files) > 0):
                data_x = []
                data_y = []   
                min_len = 1000000
                for file in files:
                    file_dir = os.path.join(current_dir, file)
                    reward, timestep = load_console(file_dir, 'timesteps', 'reward', args.window[i], mode=args.mode[i])

                    if (len(reward) < min_len):
                        min_len = len(reward)

                    data_y.append(reward)

                data_x = np.array(timestep[:min_len])

                new_data_y = []
                for data in data_y:
                    new_data_y.append(data[:min_len])

                data_y_mean = np.mean(new_data_y, axis=0)
                data_y_std = np.std(new_data_y, axis=0)

                color_idx += 1

                ax[x, y].plot(data_x, data_y_mean, linestyle='solid', color=cmap.colors[color_idx], label=label)
                ax[x, y].fill_between(data_x, data_y_mean - 2.0 * data_y_std, data_y_mean + 2.0 * data_y_std, color=cmap.colors[color_idx], alpha=0.35)
                if args.domain is not None:
                    ax[x, y].set_title(args.domain[i], fontsize=args.text + 2)
                else:
                    ax[x, y].set_title(folder, fontsize=args.text + 2)
                ax[x, y].set_ylabel('Rewards', fontsize=args.text + 2)


    if args.sarsop is not None:
        ax[x, y].axhline(y=args.sarsop[i], linestyle='dashed', color='b', label='SARSOP')

    if args.random is not None:
        ax[x, y].axhline(y=args.random[i], linestyle='dashed', color='k', label='Random')

    ax[x, y].set_xlabel('Time Steps', fontsize=args.text + 2)

handles, labels = ax[x, y].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor = (0.5,0), fontsize=args.text)

fig.tight_layout()       
plt.show()