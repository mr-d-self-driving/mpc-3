import casadi as cd
import os

from solvers.mpcc_colloc import build_solver as mpcc_colloc
from solvers.mpcc_rk4 import build_solver as mpcc_rk4

solver = mpcc_colloc
ipopt_solver = 'ma57'
solve_method = 'colloc'

anim_save_file = os.path.join(os.getcwd(), 'out', 'mpcc_' + solve_method +'.gif')
pred_csv = os.path.join(os.getcwd(), 'out/pred.csv')
true_csv = os.path.join(os.getcwd(), 'out/true.csv')

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .04 # time-step
e = 0.1 # epsilon (value for when solving stops)

num_targets_final = 1

# # 5th-order
# xs, ys = 0, 0
# xt, yt = 3.3, 2
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/3, 0, 0, 0]
# xpts = [xs] + [.5, 2] + [xt]
# ypts = [ys] + [1, 3] + [yt]
# order = 5

# # 5th-order
# xs, ys = 0, 0
# xt, yt = 3, 3
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/2, 0, 0, 0]
# xpts = [xs] + [1, 2] + [xt]
# ypts = [ys] + [2, 2.5] + [yt]
# order = 5

# # 3rd-order
# xs, ys = -0.26, 0
# xt, yt = 2, 3
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/2, 0, 0, 0]
# xpts = [xs] + [0, 1] + [xt]
# ypts = [ys] + [1, 2] + [yt]
# order = 3

# 3rd-order
xs, ys = -0.3, 0
xt, yt = 2, 3
        # [x, y, phi, delta, vx, theta]
init_ts = [xs, ys, 2*cd.pi/3, 0, 0, 0]
xpts = [xs] + [0, 1] + [xt]
ypts = [ys] + [1.5, 1.75] + [yt]
order = 3

# # 1st-order
# xs, ys = 0, 0
# xt, yt = 2, 2 
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, cd.pi/4, 0, 0, 0]
# xpts = [xs] + [1] + [xt]
# ypts = [ys] + [1] + [yt]
# order = 1

curve_dict = {'init_ts': init_ts,
              'xpts': xpts,
              'ypts': ypts,
              'order': 3}