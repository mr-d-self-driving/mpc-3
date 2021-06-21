from casadi import *
from mpcc_loss import gen_cost_func

def build_solver(init_ts):
    T = 10. # Time horizon
    N = 40 # number of control intervals

    D = 1 # inter-axle distance

    xt = SX.sym('xt') # target x
    yt = SX.sym('yt') # target y

    # dt = xt/T

    x = SX.sym('x')
    y = SX.sym('y')
    psi = SX.sym('psi')
    delta = SX.sym('delta')
    vx = SX.sym('vx')
    theta = SX.sym('theta')

    z = vertcat(x, y, psi, delta, vx, theta)

    alphaux = SX.sym('alphaux')
    aux = SX.sym('aux')
    dt = SX.sym('dt')

    u = vertcat(alphaux, aux, dt)

    zdot = vertcat(vx*cos(psi), vx*sin(psi), (vx/D)*tan(delta), alphaux, aux, vx*dt)

    cx = SX.sym('cx', 3, 1)
    cy = SX.sym('cy', 3, 1)
    contour_cost = gen_cost_func(2)
    L = contour_cost(vertcat(x, y), theta, xt, cx, cy)

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = Function('f', [z, u], [zdot, L])
    X0 = SX.sym('X0', 6)
    U = SX.sym('U', 3)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + DT/2 * k1, U)
        k3, k3_q = f(X + DT/2 * k2, U)
        k4, k4_q = f(X + DT * k3, U)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

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
    Xk = SX.sym('X0', 6)
    w += [Xk]
    
    lbw += init_ts
    ubw += init_ts
    w0  += init_ts

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = SX.sym('U_' + str(k), 3)
        w   += [Uk]
        lbw += [-pi/8, -1, -1]
        ubw += [ pi/8,  1,  1]
        w0  += [    0,  0,  0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(k+1), 6)
        w   += [Xk]
        #        x      y    psi   delta vx theta
        lbw += [-inf, -inf, -2*pi, -pi, -2, -inf]
        ubw += [ inf,  inf,  2*pi,  pi,  1,  inf]
        w0  += [0, 0, 0, 0, 0, 0]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(xt, yt, cx, cy)}
    solver = nlpsol('solver', 'ipopt', prob)
    print(solver)

    return solver, [w0, lbw, ubw, lbg, ubg]