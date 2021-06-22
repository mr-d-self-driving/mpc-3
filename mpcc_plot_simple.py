from mpcc_solver import build_solver

import casadi as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T = 10. # Time horizon
N = 40  # number of control intervals
D = 1   # inter-axle distance

fig, (ax1, ax2) =  plt.subplots(1, 2)

xs, ys = 0, 0
xt, yt = 2, 4
        # [x, y, psi, delta, vx, theta]
init_ts = [xs, ys, 0, 0, 0, 0]

# # 3rd-order
# xpts = [xs] + [0, 1] + [xt]
# ypts = [ys] + [1, 2] + [yt]

# # 2nd-order
# xpts = [xs] + [1] + [xt]
# ypts = [ys] + [1] + [yt]

# # 1st-order
# xpts = [xs] + [1, 2] + [xt]
# ypts = [ys] + [1, 2] + [yt]

order = 2

def gen_t(pts1, pts2):
    tpts = [0]
    for i, pt in enumerate(pts1):
        if i != 0:
            dist_tmp = (pts1[i] - pts1[i-1]) ** 2 + (pts2[i] - pts2[i-1]) ** 2
            tpts += [cd.sqrt(dist_tmp) + tpts[-1]]
    maxt = tpts[-1]
    tpts = [t/maxt for t in tpts]
    return tpts

def plot():

    def sep_vals(lst):
        x_opt = lst[0::9]
        y_opt = lst[1::9]
        alphaux_opt = lst[6::9]
        aux_opt = lst[7::9]

        return x_opt, y_opt, alphaux_opt, aux_opt
    
    solver, params = build_solver(init_ts, T, N, D, order)

    w0, lbw, ubw, lbg, ubg = params

    tpts = gen_t(xpts, ypts)
    print('\n\n')
    print(tpts)
    print('\n')
    xpoly = np.polynomial.polynomial.Polynomial.fit(tpts, xpts, order)
    ypoly = np.polynomial.polynomial.Polynomial.fit(tpts, ypts, order)
    xc = list(xpoly)
    yc = list(ypoly)

    print(xc, yc)
    print('\n\n')

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=cd.vertcat(xt, yt, xc, yc))
    w_opt = sol['x'].full().flatten()

    x_opt, y_opt, alphaux_opt, aux_opt = sep_vals(w_opt)

    tgrid = [T/N*k for k in range(N+1)]

    x_diff = [xt - x for x in x_opt]
    y_diff = [yt - y for y in y_opt]

    ax1.set_ylim([-5, 5])
    ax1.plot(tgrid, x_diff, '-', color='gray')
    ax1.plot(tgrid, y_diff, '-', color='black')
    ax1.step(tgrid, [None] + list(aux_opt), '-.', color='green')
    ax1.step(tgrid, [None] + list(alphaux_opt), '-.', color='blue')

    ax1.legend(['xt - x','yt - y', 'a_x^u', 'alpha_x^u'])
    ax1.set_xlabel('Time horizon')
    ax1.grid()
    
    # plot curve
    ax2.scatter(xpts, ypts, color='grey', s=15)
    tplt = np.linspace(0, 1)
    xplt = xpoly(tplt)
    yplt = ypoly(tplt)
    ax2.plot(xplt, yplt, '-.', color='grey')

    ax2.set_ylim([-5, 5])
    ax2.set_xlim([-5, 5])
    ax2.set_xlabel('x')

    ax2.plot(x_opt, y_opt, '-', color='green', alpha=0.4)
    ax2.plot([x_opt[0]], [y_opt[0]], marker='o', color='blue')
    ax2.plot([xt], [yt], marker='x', color='blue')
    ax2.grid()

    plt.show()

plot()