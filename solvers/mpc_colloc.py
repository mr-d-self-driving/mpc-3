from mpc.utils import merge_dict

import os
import casadi as cd
import numpy as np
import matplotlib.pyplot as plt

def build_solver(init_ts, T, N, inter_axle):
    # Degree of interpolating polynomial
    d = 3

    # Get collocation points
    tau_root = np.append(0, cd.collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

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

    zdot = cd.vertcat(vx*cd.cos(phi), vx*cd.sin(phi), (vx/inter_axle)*cd.tan(delta), alphaux, aux)

    L = (x - xt)**2 + (y - yt)**2 + aux**2 + alphaux**2

    # Continuous time dynamics
    f = cd.Function('f', [z, u], [zdot, L], ['z', 'u'], ['zdot', 'L'])

    # Control discretization
    h = T/N

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
    w0 += init_ts
    coord_plot += [Xk]

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cd.SX.sym('U_' + str(k), 2)
        w += [Uk]
        lbw += [-2*cd.pi, -1]
        ubw += [ 2*cd.pi,  1]
        w0 += [0, 0]
        u_plot += [Uk]

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = cd.SX.sym('X_'+str(k)+'_'+str(j), 5)
            Xc += [Xkj]
            w += [Xkj]
            lbw += [-cd.inf, -cd.inf, -cd.inf, -cd.pi/4,  0]
            ubw += [ cd.inf,  cd.inf,  cd.inf,  cd.pi/4,  2]
            w0 += [0, 0, 0, 0, 0]

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j]*Xk
            for r in range(d): xp = xp + C[r+1,j]*Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1],Uk)
            g += [h*fj - xp]
            lbg += [0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0]

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1];

            # Add contribution to quadrature function
            J = J + B[j]*qj*h

        # New NLP variable for state at end of interval
        Xk = cd.SX.sym('X_' + str(k+1), 5)
        w += [Xk]
        lbw += [-cd.inf, -cd.inf, -cd.inf, -cd.pi/4,  0]
        ubw += [ cd.inf,  cd.inf,  cd.inf,  cd.pi/4,  2]
        w0 += [0, 0, 0, 0, 0]
        coord_plot += [Xk]

        # Add equality constraint
        g += [Xk_end-Xk]
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
    plt.imsave('out/sparsity_mpc_colloc.png', np.array(sparsity))

    # Create an NLP solver
    solver_opts = {}
    solver_opts['print_time'] = 0
    solver_opts['ipopt.print_level'] = 0
    solver_opts['ipopt.max_cpu_time'] = .5
    solver_opts['ipopt.linear_solver'] = 'ma57'

    warm_start_opts = {}
    warm_start_opts['ipopt.warm_start_init_point'] = 'yes'
    warm_start_opts['ipopt.mu_init'] = .0001
    warm_start_opts['ipopt.warm_start_bound_push'] = 1e-9
    warm_start_opts['ipopt.warm_start_bound_frac'] = 1e-9
    warm_start_opts['ipopt.warm_start_slack_bound_frac'] = 1e-9
    warm_start_opts['ipopt.warm_start_slack_bound_push'] = 1e-9
    warm_start_opts['ipopt.warm_start_mult_bound_push'] = 1e-9

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g, 'p': cd.vertcat(xt, yt)}
    solver = cd.nlpsol('solver', 'ipopt', prob, merge_dict(solver_opts, warm_start_opts));

    # solver.generate_dependencies('nlp.c')                                        
    # system('gcc -fPIC -shared -O3 nlp.c -o nlp.so')
    # solver_comp = cd.nlpsol('solver', 'ipopt', os.path.join(os.getcwd(), 'nlp.so'), merge_dict(solver_opts, warm_start_opts))

    # Function to get x and u trajectories from w
    trajectories = cd.Function('trajectories', [w], [coord_plot, u_plot], ['w'], ['x', 'u'])

    return solver, [w0[5:], lbw[5:], ubw[5:], lbg, ubg], trajectories