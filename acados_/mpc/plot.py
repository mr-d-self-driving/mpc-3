from acados_.mpc.solver import build_ocp
from acados_template import AcadosOcpSolver

import matplotlib.pyplot as plt
import numpy as np
import acados_.mpc.config as cfg

T = cfg.T
N = cfg.N

init_ts = cfg.init_ts
xt, yt = cfg.xt[0], cfg.yt[0]
target = [xt, yt] + [0]*5

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

ocp, simX, simU = build_ocp(init_ts, target, T, N)
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

print(simX)
simX_t = simX.T
a_dat, alpha_dat = simU.T
x_diff = xt - simX_t[0]
y_diff = yt - simX_t[1]

ax1.set_xlim([0, int(T)])
ax1.set_ylim([-5, 5])
x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
aux_line, = ax1.step(tgrid, np.append(np.nan, a_dat), '-.', color='green')
alphaux_line, = ax1.step(tgrid, np.append(np.nan, alpha_dat), '-.', color='blue')

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
traj, = ax2.plot(simX_t[0], simX_t[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([simX_t[0][0]], [simX_t[1][0]], marker='o', color='black')    
target_pt, = ax2.plot([xt], [yt], marker='x', color='black')
ax2.grid()

plt.show()