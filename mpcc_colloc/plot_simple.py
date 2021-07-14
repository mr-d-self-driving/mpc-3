from mpcc_collocation.solver import build_solver
from mpcc.utils import gen_t

import casadi as cd
import matplotlib.pyplot as plt
import numpy as np

# Time horizon
T = 10.
N = 40
inter_axle = .5

# 3rd-order
xs, ys = -0.3, 0
xt, yt = 2, 3
        # [x, y, phi, delta, vx, theta]
init_ts = [xs, ys, 2*cd.pi/3, 0, 0, 0]
xpts = [xs] + [0, 1] + [xt]
ypts = [ys] + [1.5, 1.75] + [yt]
order = 3

tpts = gen_t(xpts, ypts)
xpoly = np.polynomial.polynomial.Polynomial.fit(tpts, xpts, order)
ypoly = np.polynomial.polynomial.Polynomial.fit(tpts, ypts, order)
xc = list(xpoly)[::-1]
yc = list(ypoly)[::-1]

solver, params, trajectories = build_solver(init_ts, T, N, inter_axle, order, xpoly, ypoly)

w0, lbw, ubw, lbg, ubg = params
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt, xc, yc))
state_opt, u_opt = trajectories(sol['x'])
state_opt = state_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N+1)
fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))
# plt.clf()
x_diff = [xt - x for x in state_opt[0]]
y_diff = [yt - y for y in state_opt[1]]
ax1.plot(tgrid, x_diff, '-', color='gray')
ax1.plot(tgrid, y_diff, '-', color='black')
ax1.step(tgrid, np.append(np.nan, u_opt[1]), '-.', color='green')
ax1.step(tgrid, np.append(np.nan, u_opt[0]), '-.', color='blue')
plt.xlabel('t')
ax1.legend(['xt - x','yt - y', 'a_x^u', 'alpha_x^u'])
ax1.grid()
ax1.set_ylim([-5, 5])

ax2.plot(state_opt[0], state_opt[1], '-', color='green', alpha=0.4)
ax2.plot([state_opt[0][0]], [state_opt[1][0]], marker='o', color='blue')
ax2.plot([xt], [yt], marker='x', color='blue')

ax2.scatter(xpts, ypts, color='grey', s=15)
tplt = np.linspace(0, 1)
xplt = xpoly(tplt)
yplt = ypoly(tplt)
ax2.plot(xplt, yplt, '-.', color='grey')

ax2.set_ylim([-5, 5])
ax2.set_xlim([-5, 5])
ax2.set_xlabel('x')
ax2.grid()
plt.show()