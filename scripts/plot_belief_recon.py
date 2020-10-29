import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from os import path
import matplotlib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--max", type=float)
parser.add_argument("--target", type=int, default=-1)
parser.add_argument("--text", type=int, default=12)

args = parser.parse_args()

matplotlib.rc('xtick', labelsize=args.text) 
matplotlib.rc('ytick', labelsize=args.text)

step = 0
# s, b, a, s', b', empty

n_size = 14

type_belief = 'actor' if 'a' in args.file else 'critic'

with open(args.file, newline='\n') as csvfile:
    csvReader = csv.reader(csvfile)

    for row in csvReader:

        b = row[0].split(";")

        b = [float(el) for el in b]
        b = np.array(b)
        b = b.reshape(n_size + 1, n_size + 1)

        # belief
        plt.ylim(0, n_size)
        plt.xlim(0, n_size)
        plt.xlabel('Right bump position', fontsize=args.text + 2)
        plt.ylabel('Left bump position', fontsize=args.text + 2) 
        plt.imshow(b)

        plt.grid()
        plt.clim(0, args.max)
        plt.colorbar()

        saved_path = args.file + '-' + type_belief

        if not path.exists(saved_path):
            os.mkdir(saved_path)

        if step == args.target:
            plt.savefig(saved_path + '/' + str(step) + ".png", bbox_inches='tight', dpi=600)

        plt.close()
        step += 1

