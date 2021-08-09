from casadi_.mpc.utils import compute_step, get_timing

import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
import time
import casadi_.mpc.config as cfg

from casadi_.solvers.mpc_rk4 import build_solver as solver_rk4
from casadi_.solvers.mpc_colloc import build_solver as solver_colloc

plt.style.use('ggplot')

if cfg.log_simple_time:
    simple_time_csv = open(cfg.simple_time_csv, 'w')
    simple_time_writer = csv.writer(simple_time_csv)

build_solver = solver_rk4 if cfg.solve_method == 'rk4' else solver_colloc
T = cfg.T
N = cfg.N
inter_axle = cfg.inter_axle

ts = cfg.ts
e = cfg.e

rebuild_solver = False
keep_going = True
num_targets = 0

xf, yf = cfg.xf[num_targets], cfg.yf[num_targets]
init_ts = cfg.init_ts

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

solver, params, trajectories = build_solver(init_ts, T, N, inter_axle)
w0_suffix, lbw_suffix, ubw_suffix, lbg, ubg = params

def solve_mpc():
    global sol, rebuild_solver

    lbw = init_ts + lbw_suffix
    ubw = init_ts + ubw_suffix

    t0 = time.time()
    sol = solver(x0=sol['x'], lam_x0=sol['lam_x'], lam_g0=sol['lam_g'], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xf, yf))
    t1 = time.time()

    if cfg.log_simple_time:
        simple_time_writer.writerow([t1-t0])
    # cost = sol['f'].full().flatten()

    state_opt, u_opt = trajectories(sol['x'])
    state_opt = state_opt.full() # to numpy array
    u_opt = u_opt.full() # to numpy array

    return state_opt, u_opt

def gen():
    global keep_going, num_targets, xf, yf
    i = 0
    while num_targets < cfg.num_targets_final:
        if not keep_going:
            num_targets += 1
            if num_targets < cfg.num_targets_final: 
                xf, yf = cfg.xf[num_targets], cfg.yf[num_targets]
            print('number of targets reached:', num_targets)
            keep_going = True
        else: i += 1
        yield i

def update(i):
    global init_ts, keep_going, rebuild_solver

    state_opt, u_opt = solve_mpc()

    x_diff = [xf - x for x in state_opt[0]]
    y_diff = [yf - y for y in state_opt[1]]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, u_opt[1]))
    alphaux_line.set_ydata(np.append(np.nan, u_opt[0]))

    traj.set_data(state_opt[0], state_opt[1])
    curr_pt.set_data([state_opt[0][0]], [state_opt[1][0]])
    target_pt.set_data([xf], [yf])

    init = [state_opt[0][0], state_opt[1][0], state_opt[2][0], state_opt[3][0], state_opt[4][0], u_opt[0][0], u_opt[1][0]]

    init_ts = compute_step(init, ts, inter_axle)

    if abs(init_ts[0]-xf) < e and abs(init_ts[1]-yf) < e:
        print('target reached:', xf, yf)
        keep_going = False
        rebuild_solver = True
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

tgrid = [T/N*k for k in range(N+1)]

w0 = init_ts + w0_suffix
lbw = init_ts + lbw_suffix
ubw = init_ts + ubw_suffix

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xf, yf))

# cost = sol['f'].full().flatten()

w_opt = sol['x'].full().flatten()
state_opt, u_opt = trajectories(sol['x'])
state_opt = state_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

x_diff = [xf - x for x in state_opt[0]]
y_diff = [yf - y for y in state_opt[1]]

ax1.set_xlim([0, int(T)])
ax1.set_ylim([-4, 4])
x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
aux_line, = ax1.step(tgrid, np.append(np.nan, u_opt[1]), '-.', color='green')
alphaux_line, = ax1.step(tgrid, np.append(np.nan, u_opt[0]), '-.', color='blue')

amin, amax = -1, 1
alphamin, alphamax = -np.pi, np.pi

amin_pts = [amin for _ in tgrid]
amax_pts = [amax for _ in tgrid]
alphamin_pts = [alphamin for _ in tgrid]
alphamax_pts = [alphamax for _ in tgrid]

ax1.plot(tgrid, amin_pts, '--', color='green', alpha=0.3)
ax1.plot(tgrid, amax_pts, '--', color='green', alpha=0.3)
ax1.plot(tgrid, alphamin_pts, '--', color='blue', alpha=0.3)
ax1.plot(tgrid, alphamax_pts, '--', color='blue', alpha=0.3)

# ax1.set_title('Controls')
ax1.legend([r'$x_f - x$',r'$y_f - y$', r'$a$', r'$\alpha$'])
ax1.set_xlabel('Time horizon')
ax1.grid(True)

# ax2.set_title('Trajectory')
ax2.set_ylim([-5, 5])
ax2.set_xlim([-5, 5])
ax2.set_ylabel('y-axis')
ax2.set_xlabel('x-axis')

# plot curve
traj, = ax2.plot(state_opt[0], state_opt[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([state_opt[0][0]], [state_opt[1][0]], marker='o', color='black')    
target_pt, = ax2.plot([xf], [yf], marker='x', color='black')
ax2.grid(True)

writergif = animation.PillowWriter(fps=30)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
# anim.save(cfg.anim_save_file, writer=writergif)
plt.show()

if cfg.log_time:
    with open(cfg.out_log_file, 'r') as f:
        tmp = ' '.join(f.read().split('\n'))
    timing = get_timing(tmp)

    time_fl = open(cfg.time_csv, 'w')
    time_writer = csv.writer(time_fl)
    time_writer.writerow(['IN_IPOPT', 'IN_NLP'])
    for t in timing:
        time_writer.writerow(list(t))
    time_fl.close()

if cfg.log_simple_time:
    simple_time_csv.close()