from mpcc.utils import gen_t

import casadi as cd
import matplotlib.pyplot as plt
import numpy as np

import mpcc.config as cfg

build_solver = cfg.solver
T = cfg.T
N = cfg.N
inter_axle = cfg.inter_axle

curve = cfg.curve_dict
xpts, ypts = curve['xpts'], curve['ypts']
order = curve['order']
xs, ys = xpts[0], ypts[0]
xt, yt = xpts[-1], ypts[-1]
init_ts = curve['init_ts']

tpts = gen_t(xpts, ypts)
xpoly = np.polynomial.polynomial.Polynomial.fit(tpts, xpts, order)
ypoly = np.polynomial.polynomial.Polynomial.fit(tpts, ypts, order)
cx = list(xpoly)[::-1]
cy = list(ypoly)[::-1]

solver, params, trajectories = build_solver(init_ts, T, N, inter_axle, order, xpoly, ypoly)

w0, lbw, ubw, lbg, ubg = params
w0 = init_ts + w0
lbw = init_ts + lbw
ubw = init_ts + ubw

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt, cx, cy))

state_opt, u_opt = trajectories(sol['x'])
state_opt = state_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N+1)
fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))
# plt.clf()
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
ax2.scatter(xpts, ypts, color='grey', s=15)
tplt = np.linspace(0, 1)
xplt = xpoly(tplt)
yplt = ypoly(tplt)
ax2.plot(xplt, yplt, '-.', color='grey')

traj, = ax2.plot(state_opt[0], state_opt[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([state_opt[0][0]], [state_opt[1][0]], marker='o', color='black')    
target_pt, = ax2.plot([xt], [yt], marker='x', color='black')
ax2.grid()

plt.show()