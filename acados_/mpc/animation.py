from acados_.mpc.solver import build_ocp
from acados_template import AcadosOcpSolver, AcadosSimSolver

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import acados_.mpc.config as cfg

plt.style.use('ggplot')

T = cfg.T
N = cfg.N
D = cfg.D

init_ts = cfg.init_ts
xc, yc = init_ts[0], init_ts[1]
xf, yf = cfg.xf[0], cfg.yf[0]
target = [xf, yf] + [0]*5

e = cfg.e

keep_going = True
num_targets = 0

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

def solve_mpc():

    for i in range(N):
        yref = np.array([xf, yf, 0, 0, 0, 0, 0])
        ocp_solver.set(i, 'yref', yref)
    ocp_solver.set(N, 'yref', np.array([xf, yf, 0, 0, 0]))

    status = ocp_solver.solve()
    if status != 0:
        ocp_solver.print_statistics()
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
    
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, 'x')
        simU[i,:] = ocp_solver.get(i, 'u')
    simX[N,:] = ocp_solver.get(N, 'x')

    cost = ocp_solver.get_cost()
    print('cost', cost)

    x0 = simX[1]
    ocp_solver.set(0, 'lbx', x0)
    ocp_solver.set(0, 'ubx', x0)
    print('x0', x0)

    return simX, simU

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < cfg.num_targets_final:
        if not keep_going:
            num_targets += 1
            if num_targets < cfg.num_targets_final: 
                global xf, yf, target
                xf, yf = cfg.xf[num_targets], cfg.yf[num_targets]
                target = [xf, yf] + [0]*5
            else: break
            print('number of targets reached:', num_targets)
            keep_going = True
        else: i += 1
        yield i

def update(i):
    global init_ts, keep_going

    simX, simU = solve_mpc()

    simX_t = simX.T
    a_dat, alpha_dat = simU.T
    x_diff = xf - simX_t[0]
    y_diff = yf - simX_t[1]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, a_dat))
    alphaux_line.set_ydata(np.append(np.nan, alpha_dat))

    traj.set_data(simX_t[0], simX_t[1])
    xc, yc = simX[0][0], simX[0][1]
    curr_pt.set_data([xc], [yc])
    target_pt.set_data([xf], [yf])

    if abs(xc-xf) < e and abs(yc-yf) < e:
        keep_going = False
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

tgrid = [T/N*k for k in range(N+1)]

ocp, simX, simU = build_ocp(init_ts, target, T, N, D, cfg.code_export_dir)
ocp_solver = AcadosOcpSolver(ocp, json_file=cfg.json_path)

# get solution
simX, simU = solve_mpc()

simX_t = simX.T
a_dat, alpha_dat = simU.T
x_diff = xf - simX_t[0]
y_diff = yf - simX_t[1]

ax1.set_xlim([0, int(T)])
ax1.set_ylim([-4, 4])
x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
aux_line, = ax1.step(tgrid, np.append(np.nan, a_dat), '-.', color='green')
alphaux_line, = ax1.step(tgrid, np.append(np.nan, alpha_dat), '-.', color='blue')

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

ax2.set_ylim([-5, 5])
ax2.set_xlim([-5, 5])
ax2.set_ylabel('y-axis')
ax2.set_xlabel('x-axis')

# plot curve
traj, = ax2.plot(simX_t[0], simX_t[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([xc], [yc], marker='o', color='black')
target_pt, = ax2.plot([xf], [yf], marker='x', color='black')
ax2.grid(True)

writergif = animation.PillowWriter(fps=10)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
anim.save(cfg.anim_save_file, writer=writergif)
# plt.show()
