import os
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
D = 0.5

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
code_export_dir = os.path.join(curr_path, 'mpcc_acados_generated_code')
json_path = 'mpcc_acados_ocp.json'

os.makedirs(out_path, exist_ok=True)
os.makedirs(code_export_dir, exist_ok=True)

anim_save_file = os.path.join(out_path, '3_acados_mpcc.gif')

ts = .04
e = 0.1

# 5th-order
curve_1 = {'init_ts': np.array([0, 0, np.pi/3, 0, 0, 0]),
           'xpts': [0, .5, 2, 3.3],
           'ypts': [0, 1, 3, 2],
           'order': 5}
          
curve_2 = {'init_ts': np.array([3.29, 2.09, -1.66, 0, 0, 0]),
           'xpts': [3.3, 2.7, 2, 3],
           'ypts': [2, .5, -1, -2],
           'order': 5}

curve_3 = {'init_ts': np.array([3, -2, 0, 0, 0, 0]),
           'xpts': [3, 3.5, 4, 3.5, 1],
           'ypts': [-2, -2.5, -3.5, -4.5, -4.5],
           'order': 5}

curve_4 = {'init_ts': np.array([1, -4.5, 3*np.pi/4, 0, 0, 0]),
           'xpts': [1, 0, -1.5, -3, -2.5],
           'ypts': [-4.5, -3.5, -2.5, -1.5, .5],
           'order': 5}

curve_5 = {'init_ts': np.array([-2.5, .5, np.pi/4, 0, 0, 0]),
           'xpts': [-2.5, -1.5, 0, 3, 2, 0],
           'ypts': [.5, 1.25, 2, 0, -1, 0],
           'order': 5}

curves_lst = [curve_5] # curve_1, curve_2, curve_3, curve_4, 
num_targets_final = len(curves_lst)