import os
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
D = 0.5

ts = .04
e = 0.1
num_targets_final = 1

xt, yt = [2.0], [3.0]
init_ts = np.array([0.0, 0.0, np.pi/4, 0.0, 0.0])

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
os.makedirs(out_path, exist_ok=True)

anim_save_file = os.path.join(out_path, 'acados_mpc.gif')