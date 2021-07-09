from mpcc.mpcc_loss import gen_cost_func

import casadi as cd
import random as rd
import numpy as np

def build_solver(init_ts, T, N, D, order, xpoly, ypoly):
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

    x = cd.SX.sym('x')
    y = cd.SX.sym('y')
    phi = cd.SX.sym('phi')
    delta = cd.SX.sym('delta')
    vx = cd.SX.sym('vx')
    theta = cd.SX.sym('theta')

    z = cd.vertcat(x, y, phi, delta, vx, theta)

    alphaux = cd.SX.sym('alphaux')
    aux = cd.SX.sym('aux')
    dt = cd.SX.sym('dt')

    u = cd.vertcat(alphaux, aux, dt)

    zdot = cd.vertcat(vx*cd.cos(phi), vx*cd.sin(phi), (vx/D)*cd.tan(delta), alphaux, aux, vx*dt)

    xc = cd.SX.sym('xc', order + 1, 1)
    yc = cd.SX.sym('yc', order + 1, 1)
    contour_cost = gen_cost_func(order)

    L = contour_cost(pos=cd.vertcat(x, y), a=aux, alpha=alphaux, dt=dt, t=theta, t_dest=1.0, cx=xc, cy=yc)['cost']

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
    x_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = cd.SX.sym('X0', 6)
    w += [Xk]
    
    lbw += init_ts
    ubw += init_ts
    w0  += init_ts

    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cd.MX.sym('U_' + str(k))
        w.append(Uk)
        lbw.append([-1])
        ubw.append([1])
        w0.append([0])
        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = cd.MX.sym('X_'+str(k)+'_'+str(j), 2)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-0.25, -np.inf])
            ubw.append([np.inf,  np.inf])
            w0.append([0, 0])

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0,j]*Xk
            for r in range(d): xp = xp + C[r+1,j]*Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1],Uk)
            g.append(h*fj - xp)
            lbg.append([0, 0])
            ubg.append([0, 0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1];

            # Add contribution to quadrature function
            J = J + B[j]*qj*h

        # New NLP variable for state at end of interval
        Xk = cd.MX.sym('X_' + str(k+1), 6)
        w.append(Xk)
        lbw.append([-0.25, -np.inf])
        ubw.append([np.inf,  np.inf])
        w0.append([0, 0])
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.append([0, 0])
        ubg.append([0, 0])

    # Concatenate vectors
    w = cd.vertcat(*w)
    g = cd.vertcat(*g)
    x_plot = cd.horzcat(*x_plot)
    u_plot = cd.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    solver = cd.nlpsol('solver', 'ipopt', prob);

    return solver, [w0[6:], lbw[6:], ubw[6:], lbg, ubg]