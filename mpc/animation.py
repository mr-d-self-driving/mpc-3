from mpc.utils import compute_step

import time
import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import mpc.config as cfg

build_solver = cfg.solver
T = cfg.T
N = cfg.N
inter_axle = cfg.inter_axle

ts = cfg.ts
e = cfg.e

keep_going = True
num_targets = 0

xt, yt = cfg.xt[num_targets], cfg.yt[num_targets]
init_ts = cfg.init_ts

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

solver, params, trajectories = build_solver(init_ts, T, N, inter_axle)
_, lbw_suffix, ubw_suffix, lbg, ubg = params

def solve_mpcc():
    global sol

    lbw = init_ts + lbw_suffix
    ubw = init_ts + ubw_suffix

    sol = solver(x0=sol['x'], lam_x0=sol['lam_x'], lam_g0=sol['lam_g'], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt))

    # cost = sol['f'].full().flatten()

    state_opt, u_opt = trajectories(sol['x'])
    state_opt = state_opt.full() # to numpy array
    u_opt = u_opt.full() # to numpy array

    return state_opt, u_opt

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < cfg.num_targets_final:
        i += 1
        if not keep_going:
            num_targets += 1
            print(num_targets)
            keep_going = True
        yield i

def update(i):
    global init_ts, keep_going, xt, yt

    state_opt, u_opt = solve_mpcc()

    x_diff = [xt - x for x in state_opt[0]]
    y_diff = [yt - y for y in state_opt[1]]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, u_opt[1]))
    alphaux_line.set_ydata(np.append(np.nan, u_opt[0]))

    traj.set_data(state_opt[0], state_opt[1])
    curr_pt.set_data([state_opt[0][0]], [state_opt[1][0]])

    init = [state_opt[0][0], state_opt[1][0], state_opt[2][0], state_opt[3][0], state_opt[4][0], u_opt[0][0], u_opt[1][0]]

    init_ts = compute_step(init, ts, inter_axle)

    if abs(init_ts[0]-xt) < e and abs(init_ts[1]-yt) < e:
        keep_going = False
        xt, yt = cfg.xt[num_targets], cfg.xt[num_targets]
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

tgrid = [T/N*k for k in range(N+1)]

w0, lbw, ubw, lbg, ubg = params
w0 = init_ts + w0
lbw = init_ts + lbw
ubw = init_ts + ubw

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt))

# cost = sol['f'].full().flatten()

w_opt = sol['x'].full().flatten()
state_opt, u_opt = trajectories(sol['x'])
state_opt = state_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

x_diff = [xt - x for x in state_opt[0]]
y_diff = [yt - y for y in state_opt[1]]

ax1.set_xlim([0, int(T)])
ax1.set_ylim([-5, 5])
x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
aux_line, = ax1.step(tgrid, np.append(np.nan, u_opt[1]), '-.', color='green')
alphaux_line, = ax1.step(tgrid, np.append(np.nan, u_opt[0]), '-.', color='blue')

ax1.set_title('Controls')
ax1.legend(['xt - x','yt - y', r'$a$', r'$\alpha$'])
ax1.set_xlabel('Time horizon')
ax1.grid()

ax2.set_title('Trajectory')
ax2.set_ylim([-5, 5])
ax2.set_xlim([-5, 5])
ax2.set_ylabel('y-axis')
ax2.set_xlabel('x-axis')

# plot curve
traj, = ax2.plot(state_opt[0], state_opt[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([state_opt[0][0]], [state_opt[1][0]], marker='o', color='black')    
target_pt, = ax2.plot([xt], [yt], marker='x', color='black')
ax2.grid()

writergif = animation.PillowWriter(fps=30)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
# anim.save(cfg.anim_save_file, writer=writergif)
plt.show()
