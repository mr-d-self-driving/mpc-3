import os
import numpy as np

T = 1. # Time horizon
N = 50  # number of control intervals
D = 0.5 # inter-axle

e = 0.07

init_ts = np.array([0, 0, np.pi/4, 0, 0])
# init_ts = np.array([2, 1, np.pi/2, 0, 0])
# xf, yf = [0, -3, -2, 1, 2], [3, 0, -3, -1, 1]
xf, yf = [5], [7]
num_targets_final = len(xf)

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # sorta hacky
curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
code_export_dir = os.path.join(curr_path, 'mpc_acados_generated_code')
json_path = 'mpc_acados_ocp.json'

os.makedirs(out_path, exist_ok=True)
os.makedirs(code_export_dir, exist_ok=True)

anim_save_file = os.path.join(out_path, 'acados_mpc.gif')