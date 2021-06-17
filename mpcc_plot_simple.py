from mpcc_solver import build_solver
from casadi import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation

T = 10. # Time horizon
N = 40 # number of control intervals

fig, (ax1, ax2) =  plt.subplots(1, 2)
init_ts = [0, 0, 0, 0, 0]
target_x, target_y = 2, 4

# simple curve: y = x^2
curve_x = (lambda t: t)
curve_y = (lambda t: t**2 + 1)

def plot():

    def sep_vals(lst):
        x_opt = lst[0::8]
        y_opt = lst[1::8]
        alphaux_opt = lst[5::8]
        aux_opt = lst[6::8]
        t_opt = lst[7::8]

        return x_opt, y_opt, alphaux_opt, aux_opt, t_opt
    
    solver, params = build_solver(curve_x, curve_y, init_ts)

    w0, lbw, ubw, lbg, ubg = params

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(target_x, target_y))
    w_opt = sol['x'].full().flatten()

    x_opt, y_opt, alphaux_opt, aux_opt, t_opt = sep_vals(w_opt)

    tgrid = [T/N*k for k in range(N+1)]

    x_diff = [target_x - x for x in x_opt]
    y_diff = [target_y - y for y in y_opt]

    ax1.plot(tgrid, x_diff, '-', color='gray')
    ax1.plot(tgrid, y_diff, '-', color='black')
    ax1.step(tgrid, [None] + list(aux_opt), '-.', color='green')
    ax1.step(tgrid, [None] + list(alphaux_opt), '-.', color='blue')
    ax1.step(tgrid, [None] + list(t_opt), '-.', color='purple')

    ax1.legend(['xt - x','yt - y','a', 'alpha', 't'])
    ax1.grid()

    ax2.set_ylim([-3, 5])
    ax2.set_xlim([-3, 3])

    ax2.plot(x_opt, y_opt, '-o')
    ax2.grid()
    
    # plot curve
    x_vals = np.linspace(-3, 3, 50)
    y_vals = curve_y(x_vals)
    ax2.plot(x_vals, y_vals, '-', color='black')

    plt.show()

plot()