from acados_.mpcc.solver import build_ocp
from acados_.mpcc.utils import get_curve
from acados_template import AcadosOcpSolver

import matplotlib.pyplot as plt
import numpy as np
import acados_.mpcc.config as cfg

plt.style.use('ggplot')

T = cfg.T
N = cfg.N
D = cfg.D

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

curve = cfg.curves_lst[0]
xs, ys, xf, yf, init_ts, xpts, ypts, tpts, xpoly, ypoly, cx, cy, order = get_curve(curve)

print(cx, cy)

ocp, simX, simU = build_ocp(init_ts, cx, cy, order, T, N, D, cfg.code_export_dir)
ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
status = ocp_solver.solve()

tgrid = np.linspace(0, T, N+1)

if status != 0:
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
    raise Exception('acados returned status {}. Exiting.'.format(status))

# get solution
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

simX_t = simX.T
a_dat, alpha_dat = simU.T
x_diff = xf - simX_t[0]
y_diff = yf - simX_t[1]

ax1.set_xlim([0, int(T)])
ax1.set_ylim([-5, 5])
x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
aux_line, = ax1.step(tgrid, np.append(np.nan, a_dat), '-.', color='green')
alphaux_line, = ax1.step(tgrid, np.append(np.nan, alpha_dat), '-.', color='blue')

ax1.legend([r'$xf - x$', r'$yf - y$', r'$a$', r'$\alpha$'])
ax1.set_xlabel('Time horizon')
ax1.grid(True)

ax2.set_ylim([-5, 5])
ax2.set_xlim([-5, 5])
ax2.set_ylabel('y-axis')
ax2.set_xlabel('x-axis')

# plot curve
curve_pts = ax2.scatter(xpts, ypts, color='grey', s=15)
tplt = np.linspace(0, 1)
xplt = xpoly(tplt)
yplt = ypoly(tplt)
curve_ln, = ax2.plot(xplt, yplt, '-.', color='grey')

traj, = ax2.plot(simX_t[0], simX_t[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([simX_t[0][0]], [simX_t[1][0]], marker='o', color='black')    
target_pt, = ax2.plot([xf], [yf], marker='x', color='black')
ax2.grid(True)

plt.show()