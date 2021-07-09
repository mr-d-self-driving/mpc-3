from casadi.casadi import diff
from mpcc_collocation.solver import build_solver
from mpcc.utils import gen_t, compute_step

import time
import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
inter_axle = 0.5   # inter-axle distance

ts = .033 # time-step
e = 0.07

keep_going = True
num_targets = 0

fig, (ax1, ax2, ax3) =  plt.subplots(1, 3, figsize=(15, 5))

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

tpts = gen_t(xpts, ypts)
xpoly = np.polynomial.polynomial.Polynomial.fit(tpts, xpts, order)
ypoly = np.polynomial.polynomial.Polynomial.fit(tpts, ypts, order)
cx = list(xpoly)[::-1]
cy = list(ypoly)[::-1]

print(cx, cy)

solver, params, trajectories = build_solver(init_ts, T, N, inter_axle, order, xpoly, ypoly)

time_y = [None]
time_x = [0]

def solve_mpcc(w0_tmp=None):
    global time_y, time_x, w_opt
    w0, lbw, ubw, lbg, ubg = params

    if w0_tmp is None: w0 = w_opt
    else: w0 = w0_tmp + w0

    lbw = init_ts + lbw
    ubw = init_ts + ubw

    time_before_sol = time.time()

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt, cx, cy))

    time_after_sol = time.time()
    diff_sol = time_after_sol - time_before_sol
    time_y.append(diff_sol)
    time_x.append(time_x[-1]+1)

    # cost = sol['f'].full().flatten()
    w_opt = sol['x'].full().flatten()
    state_opt, u_opt = trajectories(sol['x'])
    state_opt = state_opt.full() # to numpy array
    u_opt = u_opt.full() # to numpy array

    return state_opt, u_opt

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < 1:
        i += 1
        if not keep_going:
            num_targets += 1
            print(num_targets)
            keep_going = True
        yield i

def update(i):
    global x_line, y_line, aux_line, alphaux_line
    global curr_pt, target_pt, traj
    global init_ts, xt, yt, num_targets, keep_going, time_x, time_y

    state_opt, u_opt = solve_mpcc()

    x_diff = [xt - x for x in state_opt[0]]
    y_diff = [yt - y for y in state_opt[1]]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, u_opt[1]))
    alphaux_line.set_ydata(np.append(np.nan, u_opt[0]))

    traj.set_data(state_opt[0], state_opt[1])
    curr_pt.set_data([state_opt[0][0]], [state_opt[1][0]])

    time_pts.set_data(time_x, time_y)

    init = [state_opt[0][0], state_opt[1][0], state_opt[2][0], state_opt[3][0], state_opt[4][0], state_opt[5][0], u_opt[0][0], u_opt[1][0], u_opt[2][0]]

    init_ts = compute_step(init, ts, inter_axle)

    if abs(init_ts[0]-xt) < e and abs(init_ts[1]-yt) < e:
        keep_going = False
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

def init_plot():
    global t_grid
    global x_line, y_line, aux_line, alphaux_line, curr_pt, target_pt, traj
    global init_ts, xt, yt, num_targets, keep_going, time_pts

    tgrid = [T/N*k for k in range(N+1)]

    state_opt, u_opt = solve_mpcc(w0_tmp=init_ts)

    x_diff = [xt - x for x in state_opt[0]]
    y_diff = [yt - y for y in state_opt[1]]

    ax1.set_ylim([-5, 5])
    x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
    y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
    aux_line, = ax1.step(tgrid, np.append(np.nan, u_opt[1]), '-.', color='green')
    alphaux_line, = ax1.step(tgrid, np.append(np.nan, u_opt[0]), '-.', color='blue')

    ax1.legend(['xt - x','yt - y', 'a_x^u', 'alpha_x^u'])
    ax1.set_xlabel('Time horizon')
    ax1.grid()

    ax2.set_ylim([-5, 5])
    ax2.set_xlim([-5, 5])
    ax2.set_xlabel('x')

    # plot curve
    ax2.scatter(xpts, ypts, color='grey', s=15)
    tplt = np.linspace(0, 1)
    xplt = xpoly(tplt)
    yplt = ypoly(tplt)
    ax2.plot(xplt, yplt, '-.', color='grey')

    traj, = ax2.plot(state_opt[0], state_opt[1], '-', color='green', alpha=0.4)
    curr_pt, = ax2.plot([state_opt[0][0]], [state_opt[1][0]], marker='o', color='blue')    
    target_pt, = ax2.plot([xt], [yt], marker='x', color='blue')
    ax2.grid()

    ax3.set_xlim([0, 200])
    ax3.set_ylim([0, .25])
    time_pts, = ax3.plot(time_x, time_y, 'o', color='blue', lw=.2, alpha=0.3)
    ax3.grid(axis='y')

init_plot()

writergif = animation.PillowWriter(fps=30)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
anim.save('test_mpcc_colloc_3_time.gif', writer=writergif)

# plt.show()