from mpcc_solver import build_solver
from casadi import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation

T = 10. # Time horizon
N = 40 # number of control intervals

fig, (ax1, ax2) =  plt.subplots(1, 2)
init_ts = [1, 1, 0, 0, 0, 0]
target_x, target_y = 2, 2

p_degree = 2

def plot():

    def sep_vals(lst):
        x_opt = lst[0::9]
        y_opt = lst[1::9]
        theta_opt = lst[5::9]
        alphaux_opt = lst[6::9]
        aux_opt = lst[7::9]

        return x_opt, y_opt, theta_opt, alphaux_opt, aux_opt
    
    solver, params = build_solver(init_ts, T, N)

    w0, lbw, ubw, lbg, ubg = params

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(target_x, target_y, vertcat(0, 1, 0), vertcat(0, 1, 0)))
    w_opt = sol['x'].full().flatten()

    x_opt, y_opt, theta_opt, alphaux_opt, aux_opt = sep_vals(w_opt)

    tgrid = [T/N*k for k in range(N+1)]

    x_diff = [target_x - x for x in x_opt]
    y_diff = [target_y - y for y in y_opt]

    ax1.plot(tgrid, x_diff, '-', color='gray')
    ax1.plot(tgrid, y_diff, '-', color='black')
    ax1.plot(tgrid, theta_opt, '-', color='blue')
    ax1.step(tgrid, [None] + list(aux_opt), '-.', color='green')
    ax1.step(tgrid, [None] + list(alphaux_opt), '-.', color='blue')

    ax1.legend(['xt - x','yt - y', 'theta', 'a_x^u', 'alpha_x^u'])
    ax1.grid()

    ax2.set_ylim([-5, 5])
    ax2.set_xlim([-5, 5])

    ax2.plot(x_opt, y_opt, '-', color='green', alpha=0.4)
    ax2.plot([x_opt[0]], [y_opt[0]], marker='o', color='blue')
    ax2.plot([target_x], [target_y], marker='x', color='blue')
    ax2.grid()
    
    # plot curve
    # num = int(T*N)
    # x_vals = [target_x*k/num for k in range(num)]
    # y_vals = [target_y/(target_x**p_degree)*x**p_degree for x in x_vals]
    # ax2.plot(x_vals, y_vals, '-', color='black', alpha=0.4)

    plt.show()

plot()