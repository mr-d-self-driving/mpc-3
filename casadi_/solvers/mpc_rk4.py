from casadi_.mpc.utils import merge_dict
from os import system

import os
import casadi as cd
import numpy as np
import matplotlib.pyplot as plt

import casadi_.mpc.config as cfg

def build_solver(init_ts, T, N, D):
    
    xt = cd.SX.sym('xt') # target x
    yt = cd.SX.sym('yt') # target y

    x = cd.SX.sym('x')
    y = cd.SX.sym('y')
    phi = cd.SX.sym('phi')
    delta = cd.SX.sym('delta')
    vx = cd.SX.sym('vx')

    z = cd.vertcat(x, y, phi, delta, vx)

    alphaux = cd.SX.sym('alphaux')
    aux = cd.SX.sym('aux')

    u = cd.vertcat(alphaux, aux)

    zdot = cd.vertcat(vx*cd.cos(phi), vx*cd.sin(phi), (vx/D)*cd.tan(delta), alphaux, aux)

    L = (x - xt)**2 + (y - yt)**2 + aux**2 + alphaux**2

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = cd.Function('f', [z, u], [zdot, L])
    X0 = cd.SX.sym('X0', 5)
    U = cd.SX.sym('U', 2)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT/2 * k1, U)
        k3, k3_q = f(X + DT/2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)

    F = cd.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # For plotting x and u given w
    coord_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = cd.SX.sym('X0', 5)
    w += [Xk]
    lbw += init_ts
    ubw += init_ts
    w0  += init_ts
    coord_plot += [Xk]

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cd.SX.sym('U_' + str(k), 2)
        w   += [Uk]
        #       alphaux  aux
        lbw += [-cd.pi, -1]
        ubw += [ cd.pi,  1]
        w0  += [0, 0]
        u_plot += [Uk]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = cd.SX.sym('X_' + str(k+1), 5)
        w   += [Xk]
        #          x         y       phi     delta   vx
        lbw += [-cd.inf, -cd.inf, -cd.inf, -cd.pi/4,  0]
        ubw += [ cd.inf,  cd.inf,  cd.inf,  cd.pi/4,  2]
        w0  += [0, 0, 0, 0, 0]
        coord_plot += [Xk]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]
    
    # Concatenate vectors
    w = cd.vertcat(*w)
    g = cd.vertcat(*g)
    coord_plot = cd.horzcat(*coord_plot)
    u_plot = cd.horzcat(*u_plot)

    # plot sparsity
    sg = cd.sum1(g)
    sparsity = cd.jacobian(cd.jacobian(sg, w), w).sparsity()
    plt.imsave(os.path.join(cfg.out_path, 'sparsity_rk4.png'), np.array(sparsity))

    # Create an NLP solver
    solver_opts = {}
    solver_opts['print_time'] = 0
    solver_opts['ipopt.print_level'] = 0
    solver_opts['ipopt.max_cpu_time'] = .5
    solver_opts['ipopt.linear_solver'] = cfg.ipopt_solver

    warm_start_opts = {}
    warm_start_opts['ipopt.warm_start_init_point'] = 'yes'
    warm_start_opts['ipopt.mu_init'] = .0001
    warm_start_opts['ipopt.warm_start_bound_push'] = 1e-9
    warm_start_opts['ipopt.warm_start_bound_frac'] = 1e-9
    warm_start_opts['ipopt.warm_start_slack_bound_frac'] = 1e-9
    warm_start_opts['ipopt.warm_start_slack_bound_push'] = 1e-9
    warm_start_opts['ipopt.warm_start_mult_bound_push'] = 1e-9

    prob = {'f': J, 'x': w, 'g': g, 'p': cd.vertcat(xt, yt)}
    solver = cd.nlpsol('solver', 'ipopt', prob, merge_dict(solver_opts, warm_start_opts))

    if cfg.gen_compiled:
        solver.generate_dependencies('nlp.c')                                        
        system('gcc -fPIC -shared -O3 nlp.c -o nlp.so')
    if cfg.use_compiled:
        solver = cd.nlpsol('solver', 'ipopt', cfg.compiled_path, merge_dict(solver_opts, warm_start_opts))

    # Function to get x and u trajectories from w
    trajectories = cd.Function('trajectories', [w], [coord_plot, u_plot], ['w'], ['x', 'u'])

    return solver, [w0[5:], lbw[5:], ubw[5:], lbg, ubg], trajectories