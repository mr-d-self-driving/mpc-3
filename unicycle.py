import random
import time
from casadi import *

T = 10. # Time horizon
N = 40 # number of control intervals

def nlp(x_init, y_init, theta_init, v_init, omega_init, xt, yt):

    xs = MX.sym('xs')
    ys = MX.sym('ys')
    theta = MX.sym('theta')
    v = MX.sym('v')
    w = MX.sym('w')

    x = vertcat(xs, ys, theta, v, w)

    a = MX.sym('a')
    alpha = MX.sym('alpha')
    u = vertcat(a, alpha)

    xdot = vertcat(v*cos(theta), v*sin(theta), w, a, alpha)

    L = (xs-xt)**2 + (ys-yt)**2 + a**2 + alpha**2

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = T/N/M
    f = Function('f', [x, u], [xdot, L])
    X0 = MX.sym('X0', 5)
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
    Xk = MX.sym('X0', 5)
    w += [Xk]
    lbw += [x_init, y_init, theta_init, v_init, omega_init]
    ubw += [x_init, y_init, theta_init, v_init, omega_init]
    w0  += [x_init, y_init, theta_init, v_init, omega_init]

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
        Xk = MX.sym('X_' + str(k+1), 5)
        w   += [Xk]
        lbw += [-inf, -inf, -2*pi, -1, -1]
        ubw += [ inf,  inf,  2*pi, 1,  1]
        w0  += [0, 0, 0, 0, 0]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0]

    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
    solver = nlpsol('solver', 'ipopt', prob);

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()

    return w_opt

# Plot the solution

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

def solved_vals(x_init, y_init, theta_init, v_init, omega_init, xt, yt):

    def sep_vals(lst):
        x_opt = lst[0::7]
        y_opt = lst[1::7]
        theta_opt = lst[2::7]
        v_opt = lst[3::7]
        omega_opt = lst[4::7]
        a_opt = lst[5::7]
        alpha_opt = lst[6::7]

        return x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt
    
    w_opt = nlp(x_init, y_init, theta_init, v_init, omega_init, xt, yt)
    x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = sep_vals(w_opt)

    return x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt
    
def plot(x_init, y_init, theta_init, v_init, omega_init, xt, yt, dt=0.0):

    x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = solved_vals(x_init, y_init, theta_init, v_init, omega_init, xt, yt)

    x_diff = [xt - x for x in x_opt]
    y_diff = [yt - y for y in y_opt]
    tgrid = [T/N*k for k in range(N+1)]

    ax1.set_ylim([-2.0, 2.0])

    ax1.plot(tgrid, x_diff, '-', color='gray')
    ax1.plot(tgrid, y_diff, '-', color='black')
    ax1.step(tgrid, vertcat(DM.nan(1), a_opt), '-.', color='green')
    ax1.step(tgrid, vertcat(DM.nan(1), alpha_opt), '-.', color='blue')

    ax1.legend(['xt - x','yt - y','a', 'alpha'])
    ax1.grid()

    ax2.set_ylim([-2, 2])
    ax2.set_xlim([-2, 2])

    ax2.plot(x_opt, y_opt, '-', color='green', alpha=0.3)
    ax2.grid()  

    ax2.plot([x_opt[0]], [y_opt[0]], marker='o', color='blue')
    ax2.plot([xt], [yt], marker='x', color='blue')

    # Plot dt time step
    v_ts = v_opt[0] + a_opt[0]*dt
    omega_ts = omega_opt[0] + alpha_opt[0]*dt

    ds = v_opt[0]*dt + (1/2)*a_opt[0]*(dt**2)
    dtheta = omega_opt[0]*dt + (1/2)*alpha_opt[0]*(dt**2)

    theta_ts = theta_opt[0] + dtheta
    dx, dy = ds*cos(theta_ts), ds*sin(theta_ts)

    print('dx', dx, 'dy', dy, 'dtheta', dtheta, 'theta', theta_ts)

    x_ts, y_ts = x_opt[0] + dx, y_opt[0] + dy

    ax2.plot([x_opt[0], x_ts], [y_opt[0], y_ts], '-', color='black')
    plt.draw()

    return x_ts, y_ts, theta_ts, v_ts, omega_ts

# def onclick(event):
#     if event.inaxes == ax2:

#         ax1.cla()
#         ax2.cla()
#         xt, yt = event.xdata, event.ydata
#         print('x = %f, y = %f'%(xt, yt))

#         plot(0, 1, xt, yt)

#         ax2.plot([xt], [yt], marker='x')
#         plt.draw()

# Initial point
x_prev_ts, y_prev_ts, theta_prev_ts, v_prev_ts, omega_prev_ts = 0, 1, 0, 0, 0
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

target_x, target_y = -1.0, -0.75
dt = 0.25

e = 0.07

while True:
    ax1.cla()
    ax2.cla()
    print(x_prev_ts, y_prev_ts, theta_prev_ts, v_prev_ts, omega_prev_ts)
    
    x_prev_ts, y_prev_ts, theta_prev_ts, v_prev_ts, omega_prev_ts = plot(x_prev_ts, y_prev_ts, theta_prev_ts, v_prev_ts, omega_prev_ts, target_x, target_y, dt=dt)
    plt.pause(dt)
    if abs(x_prev_ts-target_x) < e and abs(y_prev_ts-target_y) < e:
        target_x, target_y = float(random.randint(-200, 200))/100, float(random.randint(-200, 200))/100

plt.show()