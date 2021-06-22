import casadi as cd
from mpcc_loss import gen_cost_func

def build_solver(init_ts, T, N, D, order):

    xt = cd.SX.sym('xt') # target x
    yt = cd.SX.sym('yt') # target y

    x = cd.SX.sym('x')
    y = cd.SX.sym('y')
    psi = cd.SX.sym('psi')
    delta = cd.SX.sym('delta')
    vx = cd.SX.sym('vx')
    theta = cd.SX.sym('theta')

    z = cd.vertcat(x, y, psi, delta, vx, theta)

    alphaux = cd.SX.sym('alphaux')
    aux = cd.SX.sym('aux')
    dt = cd.SX.sym('dt')

    u = cd.vertcat(alphaux, aux, dt)

    zdot = cd.vertcat(vx*cd.cos(psi), vx*cd.sin(psi), (vx/D)*cd.tan(delta), alphaux, aux, vx*dt)

    cx = cd.SX.sym('cx', order + 1, 1)
    cy = cd.SX.sym('cy', order + 1, 1)
    contour_cost = gen_cost_func(order)
    L = contour_cost(cd.vertcat(x, y), aux, alphaux, dt, theta, xt, cx, cy)

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = cd.Function('f', [z, u], [zdot, L])
    X0 = cd.SX.sym('X0', 6)
    U = cd.SX.sym('U', 3)
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

    # "Lift" initial conditions
    Xk = cd.SX.sym('X0', 6)
    w += [Xk]
    
    lbw += init_ts
    ubw += init_ts
    w0  += init_ts

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cd.SX.sym('U_' + str(k), 3)
        w   += [Uk]
        lbw += [-cd.pi, -1, -1]
        ubw += [ cd.pi,  1,  1]
        w0  += [     0,  0,  0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = cd.SX.sym('X_' + str(k+1), 6)
        w   += [Xk]
        #          x         y      psi    delta  vx theta
        lbw += [-cd.inf, -cd.inf, -cd.pi, -cd.pi, -1, 0]
        ubw += [ cd.inf,  cd.inf,  cd.pi,  cd.pi,  1, 1]
        w0  += [0, 0, 0, 0, 0, 0]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

    # Create an NLP solver
    prob = {'f': J, 'x': cd.vertcat(*w), 'g': cd.vertcat(*g), 'p': cd.vertcat(xt, yt, cx, cy)}
    solver = cd.nlpsol('solver', 'ipopt', prob)
    print(solver)

    return solver, [w0, lbw, ubw, lbg, ubg]