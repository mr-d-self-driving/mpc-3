from casadi import *

def build_solver(curve_x, curve_y, init_ts):
    T = 10. # Time horizon
    N = 40 # number of control intervals

    D = 1 # inter-axle distance
    c1 = 1 # scaling factor for getting to target
    c2 = 1 # scaling factor for following polynomial

    xt = MX.sym('xt')
    yt = MX.sym('yt')

    x = MX.sym('x')
    y = MX.sym('y')
    psi = MX.sym('psi')
    delta = MX.sym('delta')
    vx = MX.sym('vx')
    omega = MX.sym('omega')

    z = vertcat(x, y, psi, delta, vx, omega)

    alphaux = MX.sym('alphaux')
    aux = MX.sym('aux')

    u = vertcat(alphaux, aux)

    zdot = vertcat(vx*cos(psi), vx*sin(psi), (vx/D)*tan(delta), alphaux, aux, omega)

    L = c1*(x-xt)**2 + c1*(y-yt)**2 + aux**2 + alphaux**2 + c2*omega**2

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = Function('f', [z, u], [zdot, L])
    X0 = MX.sym('X0', 6)
    U = MX.sym('U', 2)
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
    Xk = MX.sym('X0', 6)
    w += [Xk]
    
    lbw += init_ts
    ubw += init_ts
    w0  += init_ts

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k), 2)
        w   += [Uk]
        lbw += [-1, -1]
        ubw += [ 1,  1]
        w0  += [ 0,  0]

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str(k+1), 6)
        w   += [Xk]
        #        x      y    psi   delta  vx  omega
        lbw += [-inf, -inf, -2*pi, -2*pi, -1, 0]
        ubw += [ inf,  inf,  2*pi,  2*pi,  1, 1]
        w0  += [0, 0, 0, 0, 0, 0]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(xt, yt)}
    solver = nlpsol('solver', 'ipopt', prob)

    return solver, [w0, lbw, ubw, lbg, ubg]