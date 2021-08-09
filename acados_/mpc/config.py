import os
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
D = 0.5 # inter-axle

ts = 0.08
e = 0.07

init_ts = np.array([0, 0, np.pi/4, 0, 0])
xf, yf = [2], [3]

# init_ts = np.array([2, 1, np.pi/2, 0, 0])
# xf, yf = [0, -3, -2, 1, 2], [3, 0, -3, -2, 1]
num_targets_final = len(xf)

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # sorta hacky
curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out_mpc')
code_export_dir = os.path.join(curr_path, 'mpc_acados_generated_code')
json_path = 'mpc_acados_ocp.json'

os.makedirs(out_path, exist_ok=True)
os.makedirs(code_export_dir, exist_ok=True)
os.makedirs(os.path.join(out_path, 'time_simple'), exist_ok=True)

log_simple_time = True
simple_time_csv = os.path.join(out_path, 'time_simple', 'simple_time.csv')

anim_save_file = os.path.join(out_path, 'acados_mpc.gif')