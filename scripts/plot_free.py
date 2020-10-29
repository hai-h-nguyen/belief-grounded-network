import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import argparse
from plot_utils import *

# colors = sns.color_palette("bright", 10)
mpl.rcParams['pdf.fonttype'] = 42
cmap = plt.get_cmap('tab10')

parser = argparse.ArgumentParser()
parser.add_argument('--abcb', nargs='+', default=None, type=str)
parser.add_argument('--ahch', nargs='+', default=None, type=str)
parser.add_argument('--bgn', nargs='+', default=None, type=str)
parser.add_argument('--ahcb', nargs='+', default=None, type=str)
parser.add_argument('--ahcs', nargs='+', default=None, type=str)
parser.add_argument('--sarsop', default=None, type=float)
parser.add_argument('--random', default=None, type=float)
parser.add_argument('--window', default=None, type=int)
parser.add_argument('--bc', default=None, nargs='+', type=str)

# For debugs
parser.add_argument('--d1', default=None, nargs='+', type=str)
parser.add_argument('--d2', default=None, nargs='+', type=str)
parser.add_argument('--d3', default=None, nargs='+', type=str)
parser.add_argument('--d4', default=None, nargs='+', type=str)
parser.add_argument('--d5', default=None, nargs='+', type=str)
parser.add_argument('--d6', default=None, nargs='+', type=str)
parser.add_argument('--mode', default='training', type=str)

args = parser.parse_args()
color_idx = 0
label_idx = 0

folders = [args.abcb, args.ahch, args.bgn, args.ahcb, args.ahcs, args.d1, args.d2, args.d3, args.d4, args.d5, args.d6]
labels = ['Ab-Cb', 'Ah-Ch', 'BGN', 'Ah-Cb', 'Ah-Cs', 'debug-1', 'debug-2', 'debug-3', 'debug-4', 'debug-5', 'debug-6']

for folder in folders:
	return_x = []
	return_y = [] 
	cut_point = None
	if folder is not None:
		for file in folder:
			if 'cut' in file:
				parser = file.split(':')
				cut_point = int(parser[1])
			else:
				reward, timestep = load_console_cut(file, 'timesteps', 'reward', args.window, cut_point, args.mode)
				return_y.append(reward)

		return_x = np.array(timestep)
		return_x = return_x.reshape(-1, )
		return_y = np.array(return_y)

		data_mean = np.mean(return_y, axis=0)
		data_std = np.std(return_y, axis=0)

		plt.plot(return_x, data_mean, linestyle='solid', color=cmap.colors[color_idx], label=labels[label_idx] + ' ' + args.mode)
		plt.fill_between(return_x, data_mean - data_std, data_mean + data_std, color=cmap.colors[color_idx], alpha=0.2)

		return_x = []
		return_y = [] 
		cut_point = None
		color_idx += 1
	label_idx += 1

if args.sarsop is not None:
	plt.axhline(y=args.sarsop, linestyle='dashed', color='b', label='SARSOP')

if args.random is not None:
	plt.axhline(y=args.random, linestyle='dashed', color='k', label='Random')


if args.bc is not None:
	rewards = []

	for file in args.bc:
		if 'cut' in file:
			parser = file.split(':')
			cut_point = int(parser[1])
		else:		
			reward = load_bc(file, cut_point)
			rewards.append(reward)

	rewards = np.array(rewards)
	rewards_mean = np.mean(rewards, axis=0)
	rewards_std = np.std(rewards, axis=0)

	x = np.linspace(0, timestep[-1], len(rewards_mean))
	plt.errorbar(x, rewards_mean, yerr=rewards_std, fmt='-o', color='r', label='BC')


plt.grid()      
plt.xlabel('Time Steps')
plt.ylabel('Rewards')
plt.legend()
plt.show()


