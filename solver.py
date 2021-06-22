import casadi as cd

T = 10. # Time horizon
N = 40 # number of control intervals

def build_solver(init_ts):
    xt = cd.MX.sym('xt')
    yt = cd.MX.sym('yt')

    xs = cd.MX.sym('xs')
    ys = cd.MX.sym('ys')
    theta = cd.MX.sym('theta')
    v = cd.MX.sym('v')
    w = cd.MX.sym('w')

    x = cd.vertcat(xs, ys, theta, v, w)

    a = cd.MX.sym('a')
    alpha = cd.MX.sym('alpha')
    u = cd.vertcat(a, alpha)

    xdot = cd.vertcat(v*cd.cos(theta), v*cd.sin(theta), w, a, alpha)

    L = (xs-xt)**2 + (ys-yt)**2 + a**2 + alpha**2

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = cd.Function('f', [x, u], [xdot, L])
    X0 = cd.MX.sym('X0', 5)
    U = cd.MX.sym('U', 2)
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
    Xk = cd.MX.sym('X0', 5)
    w += [Xk]
    
    lbw += init_ts
    ubw += init_ts
    w0  += init_ts

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = cd.MX.sym('U_' + str(k), 2)
        w   += [Uk]
        lbw += [-1, -1]
        ubw += [ 1,  1]
        w0  += [ 0,  0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = cd.MX.sym('X_' + str(k+1), 5)
        w   += [Xk]
        lbw += [-cd.inf, -cd.inf, -2*cd.pi, -1, -1]
        ubw += [ cd.inf,  cd.inf,  2*cd.pi, 1,  1]
        w0  += [0, 0, 0, 0, 0]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

    # Create an NLP solver
    prob = {'f': J, 'x': cd.vertcat(*w), 'g': cd.vertcat(*g), 'p': cd.vertcat(xt, yt)}
    solver = cd.nlpsol('solver', 'ipopt', prob, {'print_time':0, 'ipopt.print_level' : 0, 'ipopt.max_cpu_time': 0.4});

    return solver, [w0[5:], lbw[5:], ubw[5:], lbg, ubg]