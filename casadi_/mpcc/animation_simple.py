from casadi_.mpcc.utils import get_curve, compute_step
from casadi_.mpcc.loss import gen_cost_func

import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import casadi_.mpcc.config as cfg

from casadi_.solvers.mpcc_rk4 import build_solver as solver_rk4
from casadi_.solvers.mpcc_colloc import build_solver as solver_colloc

plt.style.use('ggplot')

build_solver = solver_rk4 if cfg.solve_method == 'rk4' else solver_colloc
T = cfg.T
N = cfg.N
inter_axle = cfg.inter_axle

ts = cfg.ts
e = cfg.e

rebuild_solver = False
keep_going = True
num_targets = 0

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

curve = cfg.curves_lst[0]
xs, ys, xf, yf, init_ts, xpts, ypts, tpts, xpoly, ypoly, cx, cy, order = get_curve(curve)

print(cx, cy)

cost_func = gen_cost_func(order)
solver, params, trajectories = build_solver(init_ts, T, N, inter_axle, order, xpoly, ypoly)
w0_suffix, lbw_suffix, ubw_suffix, lbg, ubg = params

def solve_mpcc():
    global sol, rebuild_solver

    if rebuild_solver:
        global solver, params, trajectories, w0_suffix, lbw_suffix, ubw_suffix
        solver, params, trajectories = build_solver(init_ts, T, N, inter_axle, order, xpoly, ypoly)
        w0_suffix, lbw_suffix, ubw_suffix, _, _ = params
        print('Rebuilt solver')
        rebuild_solver = False

        w0 = init_ts + w0_suffix
        lbw = init_ts + lbw_suffix
        ubw = init_ts + ubw_suffix

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xf, yf, cx, cy))
    else:
        lbw = init_ts + lbw_suffix
        ubw = init_ts + ubw_suffix
        sol = solver(x0=sol['x'], lam_x0=sol['lam_x'], lam_g0=sol['lam_g'], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xf, yf, cx, cy))

    # cost = sol['f'].full().flatten()

    state_opt, u_opt = trajectories(sol['x'])
    state_opt = state_opt.full() # to numpy array
    u_opt = u_opt.full() # to numpy array

    return state_opt, u_opt

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < cfg.num_targets_final:
        if not keep_going:
            num_targets += 1
            if num_targets < cfg.num_targets_final:
                global xs, ys, xf, yf, init_ts, xpts, ypts, tpts, xpoly, ypoly, cx, cy, order, xplt, yplt
                curve = cfg.curves_lst[num_targets]
                xs, ys, xf, yf, init_ts, xpts, ypts, tpts, xpoly, ypoly, cx, cy, order = get_curve(curve)
                print('Updated init_ts')
                tplt = np.linspace(0, 1)
                xplt = xpoly(tplt)
                yplt = ypoly(tplt)
            print('number of targets reached:', num_targets)
            keep_going = True
        else: i += 1
        yield i

def update(i):
    global init_ts, keep_going, rebuild_solver

    state_opt, u_opt = solve_mpcc()

    x_diff = [xf - x for x in state_opt[0]]
    y_diff = [yf - y for y in state_opt[1]]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, u_opt[1]))
    alphaux_line.set_ydata(np.append(np.nan, u_opt[0]))

    traj.set_data(state_opt[0], state_opt[1])
    curr_pt.set_data([state_opt[0][0]], [state_opt[1][0]])
    target_pt.set_data([xf], [yf])

    curve_pts.set_offsets(np.c_[xpts, ypts])
    curve_ln.set_data(xplt, yplt)

    init = [state_opt[0][0], state_opt[1][0], state_opt[2][0], state_opt[3][0], state_opt[4][0], state_opt[5][0], u_opt[0][0], u_opt[1][0], u_opt[2][0]]

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

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xf, yf, cx, cy))

w_opt = sol['x'].full().flatten()
state_opt, u_opt = trajectories(sol['x'])
state_opt = state_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

x_diff = [xf - x for x in state_opt[0]]
y_diff = [yf - y for y in state_opt[1]]

ax1.set_xlim([0, int(T)])
ax1.set_ylim([-5, 5])
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

ax1.legend([r'$x_f - x$',r'$y_f - y$', r'$a$', r'$\alpha$'])
ax1.set_xlabel('Time horizon')
ax1.grid(True)

ax2.set_ylim([-6, 6])
ax2.set_xlim([-6, 6])
ax2.set_ylabel('y-axis')
ax2.set_xlabel('x-axis')

# plot curve
curve_pts = ax2.scatter(xpts, ypts, color='grey', s=15)
tplt = np.linspace(0, 1)
xplt = xpoly(tplt)
yplt = ypoly(tplt)
curve_ln, = ax2.plot(xplt, yplt, '-.', color='grey')

traj, = ax2.plot(state_opt[0], state_opt[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([state_opt[0][0]], [state_opt[1][0]], marker='o', color='black')    
target_pt, = ax2.plot([xf], [yf], marker='x', color='black')
ax2.grid(True)

writergif = animation.PillowWriter(fps=30)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
anim.save(cfg.anim_save_file, writer=writergif)
# plt.show()
