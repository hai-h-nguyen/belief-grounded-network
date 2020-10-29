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

    if curr_dir == 'abcb-bgn':
        return 'Ab-Cb + BGN'

def getColor(curr_dir):
    if curr_dir == 'ahch':
        return 2

    if curr_dir == 'bgn':
        return 5

    if curr_dir == 'ahcs':
        return 1

    if curr_dir == 'ahcb':
        return 3        

    if curr_dir == 'abcb':
        return 4

    if curr_dir == 'abcb-bgn':
        return 0   

def setLabel(label):
    if label == 'Ah-Ch':
        return 'a'

    if label == 'Ah-Ch + BGN':
        return 'b'

    if label == 'Ah-Cs':
        return 'c'

    if label == 'Ah-Cb':
        return 'e'        

    if label == 'Ab-Cb':
        return 'd'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set_style('darkgrid')
cmap = plt.get_cmap('tab10')


parser = argparse.ArgumentParser()
parser.add_argument('--folder', nargs='+', type=str)
parser.add_argument('--sarsop', nargs='+', default=None, type=float)
parser.add_argument('--random', nargs='+', default=None, type=float)
parser.add_argument('--window', nargs='+', default=10, type=int)
parser.add_argument("--text", type=int, default=14)
parser.add_argument('--labels', nargs='+', default=None, type=str)
parser.add_argument('--mode', nargs='+', default='training', type=str)
parser.add_argument('--label', nargs='+', type=str)
parser.add_argument('--file', type=str)
parser.add_argument('--legend', type=int, default=1)
parser.add_argument('--show', type=int, default=1)

args = parser.parse_args()
    
mpl.rc('xtick', labelsize=args.text) 
mpl.rc('ytick', labelsize=args.text)
stat = {}

color_idx = 0

for i in range(len(args.folder)):
    # Go to the specified folder
    folder = args.folder[i]
    color_idx = 0
    for root, dirs, _ in os.walk(folder):
        for d in dirs:
            current_dir = os.path.join(root,d)
            label = getLabel(d)
            color = cmap.colors[getColor(d)]

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

                stat[setLabel(label)] = [data_y_mean[-1], data_y_std[-1]]

                plt.plot(data_x, data_y_mean, linestyle='solid', color=color, label=label)
                plt.fill_between(data_x, data_y_mean - 1 * data_y_std, data_y_mean + 1 * data_y_std, color=color, alpha=0.15)
                if args.label is not None:
                    plt.title(args.label[i], fontsize=args.text + 2)
                else:
                    plt.title(folder, fontsize=args.text + 2)

                # if args.folder in ['hallway', 'hallway2', 'rs44', 'rs55']:
                    # print("Here")
                plt.ylabel('100-Episode Avg. Return', fontsize=args.text + 2)
                # else:
                # plt.ylabel('100-Episode Avg. Sucess Rate', fontsize=args.text + 2)

    if args.sarsop is not None:
        plt.axhline(y=args.sarsop[i], linestyle='dashed', color='b', label='SARSOP')

    if args.random is not None:
        plt.axhline(y=args.random[i], linestyle='dashed', color='k', label='Random')

    plt.xlabel('Timesteps', fontsize=args.text + 2)

    if args.folder in ['rs44']:
        plt.ylim((-63.0, 20.0))

# [ print(key , " : " , value) for (key, value) in sorted(stat.items()) ]

if args.legend == 1:
    plt.legend(fontsize=args.text)
# plt.axis('off')
plt.tight_layout()

if args.show == 0:
    plt.savefig(args.file + '.png', dpi=600)
else:
    plt.show()