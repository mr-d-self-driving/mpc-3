import casadi as cd
import os

from casadi_.solvers.mpcc_colloc import build_solver as mpcc_colloc
from casadi_.solvers.mpcc_rk4 import build_solver as mpcc_rk4

solver = mpcc_rk4
ipopt_solver = 'mumps'
solve_method = 'rk4'

curr_path = os.path.dirname(os.path.dirname(__file__)) # sorta hacky
out_path = os.path.join(curr_path, 'out')
compiled_path = os.path.join(curr_path, 'compiled')

os.makedirs(out_path, exist_ok=True)
os.makedirs(compiled_path, exist_ok=True)

gen_compiled = False
use_compiled = False
compiled_path = os.path.join(compiled_path, 'nlp.so')

anim_save_file = os.path.join(out_path, 'casadi_mpcc_' + solve_method +'.gif')
pred_csv = os.path.join(out_path, 'pred.csv')
true_csv = os.path.join(out_path, 'true.csv')

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .04 # time-step
e = 0.1 # epsilon (value for when solving stops)

num_targets_final = 1

# # 5th-order
# xs, ys = 0, 0
# xf, yf = 3.3, 2
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/3, 0, 0, 0]
# xpts = [xs] + [.5, 2] + [xf]
# ypts = [ys] + [1, 3] + [yf]
# order = 5

# # 5th-order
# xs, ys = 0, 0
# xf, yf = 3, 3
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/2, 0, 0, 0]
# xpts = [xs] + [1, 2] + [xf]
# ypts = [ys] + [2, 2.5] + [yf]
# order = 5

# 3rd-order
xs, ys = -0.26, 0
xf, yf = 2, 3
        # [x, y, phi, delta, vx, theta]
init_ts = [xs, ys, cd.pi/2, 0, 0, 0]
xpts = [xs] + [0, 1] + [xf]
ypts = [ys] + [1, 2] + [yf]
order = 3

# # 3rd-order
# xs, ys = -0.3, 0
# xf, yf = 2, 3
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, 2*cd.pi/3, 0, 0, 0]
# xpts = [xs] + [0, 1] + [xf]
# ypts = [ys] + [1.5, 1.75] + [yf]
# order = 3

# # 1st-order
# xs, ys = 0, 0
# xf, yf = 2, 2 
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/4, 0, 0, 0]
# xpts = [xs] + [1] + [xf]
# ypts = [ys] + [1] + [yf]
# order = 1

curve_dict = {'init_ts': init_ts,
              'xpts': xpts,
              'ypts': ypts,
              'order': order}