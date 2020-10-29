import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import os
from os import path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--max", type=float)
parser.add_argument("--target", type=int, default=-1)
parser.add_argument("--text", type=int, default=12)
args = parser.parse_args()

line_cnt = 0
step = 0
# s, b, a, s', b', empty

matplotlib.rc('xtick', labelsize=args.text) 
matplotlib.rc('ytick', labelsize=args.text)

n_size = 14

with open(args.file, newline='\n') as csvfile:
    csvReader = csv.reader(csvfile)

    for row in csvReader:

        line_index = line_cnt % 3

        # state 
        if (line_index == 2):
            [cart_pos, angle, lb_pos, rb_pos] = row[0].split(";")
            angle = int(angle)
            cart_pos = int(cart_pos)
            lb_pos = int(lb_pos)
            rb_pos = int(rb_pos)

            # Plotting
            # belief
            plt.ylim(0, n_size)
            plt.xlim(0, n_size)
            plt.xlabel('Right bump position', fontsize=args.text + 2)
            plt.ylabel('Left bump position', fontsize=args.text + 2) 
            plt.imshow(b)

            # Left bump
            plt.plot(lb_pos, lb_pos, 'ro', markersize=10)

            # Right bump
            plt.plot(rb_pos, rb_pos, 'go', markersize=10)

            # Cart position
            # Cart position
            if (a == 0 or a == 2):
                plt.plot(cart_pos, cart_pos, 'yx', markersize=10)
            else:
                plt.plot(cart_pos, cart_pos, 'bx', markersize=10)

            for dot in range(n_size + 1):
                if dot != cart_pos and dot != rb_pos and dot != lb_pos:
                   plt.plot(dot, dot, 'wo', markersize=5) 


            plt.grid()
            plt.clim(0, args.max)
            plt.colorbar()

            saved_path = args.file + '-' + 'true'

            if not path.exists(saved_path):
                os.mkdir(saved_path)

            if step == args.target:
                plt.savefig(saved_path + '/' + str(step) + ".png", bbox_inches='tight', dpi=600)

            plt.close() 

            step += 1

        # current belief
        elif (line_index == 0):
            b = row[0].split(";")

            b = [float(el) for el in b]
            b = np.array(b)
            b = b.reshape(n_size + 1, n_size + 1)

        # action
        elif (line_index == 1):
            a = int(row[0])

       
        line_cnt += 1

line_cnt = 0
step = 0


