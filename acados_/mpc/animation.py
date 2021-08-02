from acados_.mpc.solver import build_ocp
from acados_.mpc.utils import compute_step
from acados_template import AcadosOcpSolver, AcadosSimSolver

import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import acados_.mpc.config as cfg

T = cfg.T
N = cfg.N
D = cfg.D

xcurrent = cfg.init_ts
xt, yt = cfg.xt[0], cfg.yt[0]
target = [xt, yt] + [0]*5

ts = cfg.ts
e = cfg.e

keep_going = True
num_targets = 0

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

def solve_mpc(is_start=False):
    global xcurrent

    if not is_start:
        ocp_solver.set(0, "lbx", xcurrent)
        ocp_solver.set(0, "ubx", xcurrent)

    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    status = ocp_solver.solve()

    if status != 0:
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    # simulate system
    integrator.set("x", xcurrent)
    integrator.set("u", ocp_solver.get(0, "u"))

    status = integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    xcurrent = integrator.get("x")

    return simX, simU

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < cfg.num_targets_final:
        if not keep_going:
            num_targets += 1
            print(num_targets, '\n\n')
            keep_going = True
        else:
            i += 1
            yield i

def update(i):
    global init_ts, keep_going, xt, yt

    simX, simU = solve_mpc()

    simX_t = simX.T
    a_dat, alpha_dat = simU.T
    x_diff = xt - simX_t[0]
    y_diff = yt - simX_t[1]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, a_dat))
    alphaux_line.set_ydata(np.append(np.nan, alpha_dat))

    traj.set_data(simX_t[0], simX_t[1])
    curr_pt.set_data([simX[0][0]], [simX[0][1]])

    if abs(simX[0][0]-xt) < e and abs(simX[0][1]-yt) < e:
        keep_going = False
        xt, yt = cfg.xt[num_targets], cfg.xt[num_targets]
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

tgrid = [T/N*k for k in range(N+1)]

ocp, simX, simU = build_ocp(xcurrent, target, T, N, D)
ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp.json')

# get solution
simX, simU = solve_mpc(is_start=True)

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
curr_pt, = ax2.plot([xcurrent[0]], [xcurrent[1]], marker='o', color='black')
target_pt, = ax2.plot([xt], [yt], marker='x', color='black')
ax2.grid()

writergif = animation.PillowWriter(fps=10)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
# anim.save(cfg.anim_save_file, writer=writergif)
plt.show()
