from solver import build_solver

import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from casadi import *

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

T = 10. # Time horizon
N = 40 # number of control intervals

init_ts = [0, 1, 0, 0, 0]
target_x, target_y = -1.0, -0.75

dt = 0.2
e = 0.07

keep_going = True
num_targets = 0

solver, params = build_solver(init_ts)

def solve_mpc():

    w0, lbw, ubw, lbg, ubg = params

    w0 = init_ts + w0
    lbw = init_ts + lbw
    ubw = init_ts + ubw

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(target_x, target_y))
    w_opt = sol['x'].full().flatten()

    return w_opt

def solved_vals():
    global solver, params

    def sep_vals(lst):
        x_opt = lst[0::7]
        y_opt = lst[1::7]
        theta_opt = lst[2::7]
        v_opt = lst[3::7]
        omega_opt = lst[4::7]
        a_opt = lst[5::7]
        alpha_opt = lst[6::7]

        return [x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt]

    w_opt = solve_mpc()
    opts = sep_vals(w_opt)

    return opts

def compute_step(init): # init = [x, y, theta, v, omega, a, alpha]
    x, y, theta, v, omega, a, alpha = init
    v_ts = v + a*dt
    omega_ts = omega + alpha*dt

    ds = v*dt + (1/2)*a*(dt**2)
    dtheta = omega*dt + (1/2)*alpha*(dt**2)

    theta_ts = theta + dtheta
    dx, dy = ds*cos(theta_ts), ds*sin(theta_ts)

    x_ts, y_ts = x + dx, y + dy

    return [x_ts, y_ts, theta_ts, v_ts, omega_ts]

# tgrid doesn't need to be recomputed
tgrid = [T/N*k for k in range(N+1)]

x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = solved_vals()

x_diff = [target_x - x for x in x_opt]
y_diff = [target_y - y for y in y_opt]

ax1.set_ylim([-2.0, 2.0])

ax1.legend(['xt - x','yt - y','a', 'alpha'])
ax1.grid()

x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
a_line, = ax1.step(tgrid, [None] + list(a_opt), '-.', color='green')
alpha_line, = ax1.step(tgrid, [None] + list(alpha_opt), '-.', color='blue')

ax2.set_ylim([-2, 2])
ax2.set_xlim([-2, 2])
ax2.grid()

uni_traj, = ax2.plot(x_opt, y_opt, '-', color='green', alpha=0.3)

uni_pt, = ax2.plot([x_opt[0]], [y_opt[0]], marker='o', color='blue')
target_pt, = ax2.plot([target_x], [target_y], marker='x', color='blue')

# Plot dt time step
ts = compute_step([x_opt[0], y_opt[0], theta_opt[0], v_opt[0], omega_opt[0], a_opt[0], alpha_opt[0]])
step_traj, = ax2.plot([x_opt[0], ts[0]], [y_opt[0], ts[1]], '-', color='black')

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < 5:
        i += 1
        if not keep_going:
            num_targets += 1
            keep_going = True
        yield i

def update(i):
    global x_line, y_line, a_line, alpha_line
    global uni_pt, target_pt, step_traj, uni_traj
    global init_ts, target_x, target_y, num_targets, keep_going

    x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = solved_vals()

    x_diff = [target_x - x for x in x_opt]
    y_diff = [target_y - y for y in y_opt]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    a_line.set_ydata([None] + list(a_opt))
    alpha_line.set_ydata([None] + list(alpha_opt))

    uni_traj.set_data(x_opt, y_opt)
    uni_pt.set_data([x_opt[0]], [y_opt[0]])

    init = [x_opt[0], y_opt[0], theta_opt[0], v_opt[0], omega_opt[0], a_opt[0], alpha_opt[0]]

    ts = compute_step(init)
    step_traj.set_data([x_opt[0], ts[0]], [y_opt[0], ts[1]])
    init_ts = ts

    if abs(init_ts[0]-target_x) < e and abs(init_ts[1]-target_y) < e:
        keep_going = False
        target_x, target_y = float(random.randint(-200, 200))/100, float(random.randint(-200, 200))/100
        target_pt.set_data([target_x], [target_y])
    
    return [x_line, y_line, a_line, alpha_line, uni_traj, uni_pt, step_traj]

anim = FuncAnimation(fig, update, interval=100, frames=gen)

plt.show()