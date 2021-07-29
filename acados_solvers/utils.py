import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_car(shooting_nodes, u_max, U, X_true, X_est=None, Y_measured=None, latexify=False):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: array of maximum absolute value of u's
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    # latexify plot
    if latexify:
        params = {'backend': 'ps',
                'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
                'axes.labelsize': 10,
                'axes.titlesize': 10,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.usetex': True,
                'font.family': 'serif'
        }

        matplotlib.rcParams.update(params)

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]
    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim-N_mhe)

    plt.subplot(nx+2, 1, 1)
    plt.step(t, np.append([U[0][0]], U[:, 0]), color='r')
    plt.title('closed-loop simulation')
    plt.ylabel(r'$a$')
    plt.xlabel('$t$')
    plt.hlines(u_max[0], t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max[0], t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max[0], 1.2*u_max[0]])
    plt.grid()

    plt.subplot(nx+2, 1, 2)
    plt.step(t, np.append([U[0][1]], U[:, 1]), color='r')
    plt.ylabel(r'$\alpha$')
    plt.xlabel('$t$')
    plt.hlines(u_max[1], t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max[1], t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max[1], 1.2*u_max[1]])
    plt.grid()

    states_lables = ['$x$', '$y$', r'$\phi$', r'$\delta$', r'$v_x$']

    for i in range(nx):
        plt.subplot(nx+2, 1, i+3)
        plt.plot(t, X_true[:,i], label='true')

        if WITH_ESTIMATION:
            plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
            plt.plot(t, Y_measured[:, i], 'x', label='measured')

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    # avoid plotting when running on Travis
    if os.environ.get('ACADOS_ON_TRAVIS') is None:
        plt.show()
