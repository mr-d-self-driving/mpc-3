import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from casadi import *

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))

init_ts = [0, 1, 0, 0, 0]
target_x, target_y = -1.0, -0.75

dt = 0.2
e = 0.07

keep_going = True
num_targets = 0

T = 10. # Time horizon
N = 40 # number of control intervals

def nlp(init_ts, xt, yt):

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

def solved_vals(init_ts, xt, yt):

    def sep_vals(lst):
        x_opt = lst[0::7]
        y_opt = lst[1::7]
        theta_opt = lst[2::7]
        v_opt = lst[3::7]
        omega_opt = lst[4::7]
        a_opt = lst[5::7]
        alpha_opt = lst[6::7]

        return [x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt]
    
    w_opt = nlp(init_ts, xt, yt)
    opts = sep_vals(w_opt)

    return opts

def compute_step(init): # init = [x, y, theta, v, omega, a, alpha]
    x, y, theta, v, omega, a, alpha = init
    v_ts = v + a*dt
    omega_ts = omega + alpha*dt

    ds = v*dt + (1/2)*a*(dt**2)
    dtheta = omega*dt + (1/2)*alpha*(dt**2)

    theta_ts = theta + dtheta
    dx, dy = ds*cos(theta_ts), ds*sin(theta_ts)

    x_ts, y_ts = x + dx, y + dy

    return [x_ts, y_ts, theta_ts, v_ts, omega_ts]

# tgrid doesn't need to be recomputed
tgrid = [T/N*k for k in range(N+1)]

x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = solved_vals(init_ts, target_x, target_y)

x_diff = [target_x - x for x in x_opt]
y_diff = [target_y - y for y in y_opt]

ax1.set_ylim([-2.0, 2.0])

ax1.legend(['xt - x','yt - y','a', 'alpha'])
ax1.grid()

x_line, = ax1.plot(tgrid, x_diff, '-', color='gray')
y_line, = ax1.plot(tgrid, y_diff, '-', color='black')
a_line, = ax1.step(tgrid, [None] + list(a_opt), '-.', color='green')
alpha_line, = ax1.step(tgrid, [None] + list(alpha_opt), '-.', color='blue')

ax2.set_ylim([-2, 2])
ax2.set_xlim([-2, 2])
ax2.grid()

uni_traj, = ax2.plot(x_opt, y_opt, '-', color='green', alpha=0.3)

uni_pt, = ax2.plot([x_opt[0]], [y_opt[0]], marker='o', color='blue')
target_pt, = ax2.plot([target_x], [target_y], marker='x', color='blue')

# Plot dt time step
ts = compute_step([x_opt[0], y_opt[0], theta_opt[0], v_opt[0], omega_opt[0], a_opt[0], alpha_opt[0]])
step_traj, = ax2.plot([x_opt[0], ts[0]], [y_opt[0], ts[1]], '-', color='black')

def gen():
    global keep_going, num_targets
    i = 0
    while num_targets < 5:
        i += 1
        if not keep_going:
            num_targets += 1
            keep_going = True
        yield i

def update(i):
    global x_line, y_line, a_line, alpha_line
    global uni_pt, target_pt, step_traj, uni_traj
    global init_ts, target_x, target_y, num_targets, keep_going
    
    # # clear subplots
    # ax1.cla()
    # ax2.cla()

    x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = solved_vals(init_ts, target_x, target_y)

    x_diff = [target_x - x for x in x_opt]
    y_diff = [target_y - y for y in y_opt]

    x_line.set_ydata(x_diff)
    y_line.set_ydata(y_diff)

    a_line.set_ydata([None] + list(a_opt))
    alpha_line.set_ydata([None] + list(alpha_opt))

    uni_traj.set_data(x_opt, y_opt)
    uni_pt.set_data([x_opt[0]], [y_opt[0]])

    init = [x_opt[0], y_opt[0], theta_opt[0], v_opt[0], omega_opt[0], a_opt[0], alpha_opt[0]]

    ts = compute_step(init)
    step_traj.set_data([x_opt[0], ts[0]], [y_opt[0], ts[1]])
    init_ts = ts

    if abs(init_ts[0]-target_x) < e and abs(init_ts[1]-target_y) < e:
        keep_going = False
        target_x, target_y = float(random.randint(-200, 200))/100, float(random.randint(-200, 200))/100
        target_pt.set_data([target_x], [target_y])
    
    return [x_line, y_line, a_line, alpha_line, uni_traj, uni_pt, step_traj]

anim = FuncAnimation(fig, update, interval=100, frames=gen)

plt.show()