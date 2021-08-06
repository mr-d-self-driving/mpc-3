from acados_.mpcc.solver import build_ocp
from acados_.mpcc.utils import get_curve
from acados_template import AcadosOcpSolver

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random as rd
import acados_.mpcc.config as cfg

plt.style.use('ggplot')

T = cfg.T
N = cfg.N
D = cfg.D

curve = cfg.curves_lst[0]
xs, ys, xf, yf, init_ts, xpts, ypts, tpts, xpoly, ypoly, cx, cy, order = get_curve(curve)

print(cx, cy)

tgrid = np.linspace(0, T, N+1)
e = cfg.e

keep_going = True
rebuild_solver = False
num_targets = 0

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

deltamax = np.pi/4

def solve_mpc():
    global rebuild_solver

    if rebuild_solver:
        ocp_solver.set(0, 'lbx', init_ts)
        ocp_solver.set(0, 'ubx', init_ts)
        for k in range(1, N):
            # kf = float(k)
            # theta_tmp = kf/(N-1)
            # dtheta = 0.2

            # x_tmp, y_tmp = xpoly(theta_tmp), ypoly(theta_tmp)
            # theta_step = theta_tmp + dtheta
            # phi_tmp = np.arctan((ypoly(theta_step) - y_tmp)/(xpoly(theta_step) - x_tmp))

            # ocp_solver.set(k, 'x', np.array([xpoly(theta_tmp), ypoly(theta_tmp), phi_tmp, 0, rd.randint(0, 200)/1000., theta_tmp]))
            # ocp_solver.set(k, 'u', np.array([rd.randint(-628, 628)/1000., rd.randint(-100, 100)/1000., rd.randint(0, 100)/1000.]))

            ocp_solver.set(k, 'x', np.array([0]*6))
            ocp_solver.set(k, 'u', np.array([0]*3))

        rebuild_solver = False

    p = np.array(cx + cy)

    for i in range(N):
        ocp_solver.set(i, 'p', p)
    ocp_solver.set(N, 'p', p)

    status = ocp_solver.solve()
    if status != 0:
        ocp_solver.print_statistics()
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
    
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, 'x')
        simU[i,:] = ocp_solver.get(i, 'u')
    simX[N,:] = ocp_solver.get(N, 'x')

    # cost = ocp_solver.get_cost()
    # print('cost', cost)

    x0 = simX[1]
    ocp_solver.set(0, 'lbx', x0)
    ocp_solver.set(0, 'ubx', x0)

    return simX, simU

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

    simX, simU = solve_mpc()

    simX_t = simX.T
    a_dat, alpha_dat, dt_dat = simU.T
    x_diff = xf - simX_t[0]
    y_diff = yf - simX_t[1]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    aux_line.set_ydata(np.append(np.nan, a_dat))
    alphaux_line.set_ydata(np.append(np.nan, alpha_dat))

    traj.set_data(simX_t[0], simX_t[1])

    xs, ys = simX[0][0], simX[0][1]
    curr_pt.set_data([xs], [ys])
    target_pt.set_data([xf], [yf])

    curve_pts.set_offsets(np.c_[xpts, ypts])
    curve_ln.set_data(xplt, yplt)

    if abs(xs-xf) < e and abs(ys-yf) < e:
        keep_going = False
        rebuild_solver = True
    
    return [x_line, y_line, aux_line, alphaux_line, traj, curr_pt]

tgrid = [T/N*k for k in range(N+1)]

ocp, simX, simU = build_ocp(init_ts, order, T, N, D, cfg.code_export_dir)
p = np.array(cx + cy)
ocp.parameter_values = p
ocp_solver = AcadosOcpSolver(ocp, json_file=cfg.json_path)

# get solution
simX, simU = solve_mpc()

simX_t = simX.T
a_dat, alpha_dat, dt_dat = simU.T
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

traj, = ax2.plot(simX_t[0], simX_t[1], '-', color='green', alpha=0.4)
curr_pt, = ax2.plot([xs], [ys], marker='o', color='black')
target_pt, = ax2.plot([xf], [yf], marker='x', color='black')
ax2.grid(True)

writergif = animation.PillowWriter(fps=10)
anim = animation.FuncAnimation(fig, update, interval=100, frames=gen, save_count=3000)
# anim.save(cfg.anim_save_file, writer=writergif)
plt.show()
