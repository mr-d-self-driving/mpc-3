from mpcc_solver import build_solver
from mpcc_utils import gen_t, compute_step

import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
D = 0.5   # inter-axle distance

ts = .033 # time-step
e = 0.07

keep_going = True
num_targets = 0

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

# 5th-order
xs, ys = 0, 0
xt, yt = 3.3, 2
        # [x, y, phi, delta, vx, theta]
init_ts = [xs, ys, cd.pi/3, 0, 0, 0]
xpts = [xs] + [.5, 2] + [xt]
ypts = [ys] + [1, 3] + [yt]
order = 5

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

# # 3rd-order
# xs, ys = -0.3, 0
# xt, yt = 2, 3
#         # [x, y, phi, delta, vx, theta]
# init_ts = [xs, ys, 2*cd.pi/3, 0, 0, 0]
# xpts = [xs] + [0, 1] + [xt]
# ypts = [ys] + [1.5, 1.75] + [yt]
# order = 3

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

solver, params = build_solver(init_ts, T, N, D, order, xpoly, ypoly)

def solve_mpcc(w0_tmp=None):
    global w_opt
    w0, lbw, ubw, lbg, ubg = params

    if w0_tmp == None:
        w0 = w_opt
    else: w0 = w0_tmp + w0
    lbw = init_ts + lbw
    ubw = init_ts + ubw

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt, cx, cy))
    cost = sol['f'].full().flatten()
    # print(cost)
    w_opt = sol['x'].full().flatten()
    # print(w_opt)

    return w_opt

def solved_vals(w0_tmp=None):
    global solver, params, w_opt

    def sep_vals(lst):
        x_opt = lst[0::9]
        y_opt = lst[1::9]
        psi_opt = lst[2::9]
        delta_opt = lst[3::9]
        vx_opt = lst[4::9]
        theta_opt = lst[5::9]
        alphaux_opt = lst[6::9]
        aux_opt = lst[7::9]
        dt_opt = lst[8::9]

        return [x_opt, y_opt, psi_opt, delta_opt, vx_opt, theta_opt, alphaux_opt, aux_opt, dt_opt]
    
    w_opt = solve_mpcc(w0_tmp)
    opts = sep_vals(w_opt)

    return opts

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
    global init_ts, xt, yt, num_targets, keep_going
    global x_opt, y_opt, psi_opt, delta_opt, vx_opt, theta_opt, alphaux_opt, aux_opt, dt_opt

    x_opt, y_opt, psi_opt, delta_opt, vx_opt, theta_opt, alphaux_opt, aux_opt, dt_opt = solved_vals()

    x_diff = [xt - x for x in x_opt]
    y_diff = [yt - y for y in y_opt]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata([None] + list(aux_opt))
    alphaux_line.set_ydata([None] + list(alphaux_opt))

    traj.set_data(x_opt, y_opt)
    curr_pt.set_data([x_opt[0]], [y_opt[0]])

    init = [x_opt[0], y_opt[0], psi_opt[0], delta_opt[0], vx_opt[0], theta_opt[0], alphaux_opt[0], aux_opt[0], dt_opt[0]]

    init_ts = compute_step(init, ts, D)
    # print('\n\n')
    # print(init_ts)
    # print('\n\n')

    if abs(init_ts[0]-xt) < e and abs(init_ts[1]-yt) < e:
        keep_going = False
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

def init_plot():
    global t_grid
    global x_line, y_line, aux_line, alphaux_line, curr_pt, target_pt, traj
    global init_ts, xt, yt, num_targets, keep_going
    global x_opt, y_opt, psi_opt, delta_opt, vx_opt, theta_opt, alphaux_opt, aux_opt, dt_opt

    tgrid = [T/N*k for k in range(N+1)]

    x_opt, y_opt, psi_opt, delta_opt, vx_opt, theta_opt, alphaux_opt, aux_opt, dt_opt = solved_vals(init_ts)

    x_diff = [xt - x for x in x_opt]
    y_diff = [yt - y for y in y_opt]

    ax1.set_ylim([-5, 5])
    x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
    y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
    aux_line, = ax1.step(tgrid, [None] + list(aux_opt), '-.', color='green')
    alphaux_line, = ax1.step(tgrid, [None] + list(alphaux_opt), '-.', color='blue')

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

    traj, = ax2.plot(x_opt, y_opt, '-', color='green', alpha=0.4)
    curr_pt, = ax2.plot([x_opt[0]], [y_opt[0]], marker='o', color='blue')
    target_pt, = ax2.plot([xt], [yt], marker='x', color='blue')
    ax2.grid()

init_plot()
# TODO: delete this print
# print('\n\n w0')
# nine_elem = []
# for i, v in enumerate(w_opt):
#     if (i+1) % 9 == 0:
#         print(nine_elem + [v])
#         nine_elem = []
#     else: nine_elem.append(v)
# print(nine_elem)
# print('\n\n')

writergif = animation.PillowWriter(fps=30)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
anim.save('test_mpcc_1.gif', writer=writergif)

# plt.show()