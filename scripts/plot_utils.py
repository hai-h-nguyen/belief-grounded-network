import numpy as np 
import seaborn as sns

def rolling_average(data, window_size):
	assert data.ndim == 1
	kernel = np.ones(window_size)
	smooth_data = np.convolve(data, kernel) / np.convolve(
		np.ones_like(data), kernel
	)
	return smooth_data[: -window_size + 1]

def load_bc(file, cut_point=None):
	fp = open(file, 'r')
	token = 'BC episodes'
	rewards = []
	while True: 
		line = fp.readline() 

		if token in line:
			data = line.split()
			data = data[6].split('/')
			reward = float(data[0])
			rewards.append(reward)

		if not line: 
			break

	if (cut_point is not None):
		rewards = rewards[:cut_point]  

	fp.close()
	return np.array(rewards)

def load_console(file, token_x, token_y, window_size, mode='training'):

	rewards = []
	timesteps = []
	fp = open(file, 'r')

	while True: 
		line = fp.readline() 

		if (token_x in line) and (mode in line):
			data = line.split()
			data = data[4].split(',')
			timestep = int(data[0])
			timesteps.append(timestep)

		if (token_y in line) and (mode in line):
			data = line.split()
			data = data[6].split('/')
			reward = float(data[0])
			rewards.append(reward)
	
		if not line: 
			break

	fp.close()

	rewards = np.array(rewards)
	timesteps = np.array(timesteps)

	if window_size is not None:
		rewards = rolling_average(rewards, window_size)

	return rewards, timesteps

def load_console_cut(file, token_x, token_y, window_size, cut_point=None, mode='training'):

	rewards = []
	timesteps = []
	fp = open(file, 'r')

	while True: 
		line = fp.readline() 

		if (token_x in line) and (mode in line):
			data = line.split()
			data = data[4].split(',')
			timestep = int(data[0])
			timesteps.append(timestep)

		if (token_y in line) and (mode in line):
			data = line.split()
			data = data[6].split('/')
			reward = float(data[0])
			rewards.append(reward)
	
		if not line: 
			break

	fp.close()

	rewards = np.array(rewards)
	timesteps = np.array(timesteps)

	if (cut_point is not None):
		rewards = rewards[:cut_point]    
		timesteps = timesteps[:cut_point]

	if window_size is not None:
		rewards = rolling_average(rewards, window_size)

	return rewards, timesteps




