#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
from mpcc.mpcc_loss import gen_cost_func
import casadi as cd
import numpy as np
import matplotlib.pyplot as plt

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

# Time horizon
T = 10.
inter_axle = .5

xt = 2
yt = 3

# xc = [-0.9384644717337829, 0.33812985545835283, 2.0884644717337832, 0.5118701445416473]
# yc = [1.0667002902085416, -0.14467955068929803, 0.4332997097914584, 1.6446795506892977]
# order = 3

x = cd.SX.sym('x')
y = cd.SX.sym('y')
phi = cd.SX.sym('phi')
delta = cd.SX.sym('delta')
vx = cd.SX.sym('vx')
# theta = cd.SX.sym('theta')

z = cd.vertcat(x, y, phi, delta, vx)

alphaux = cd.SX.sym('alphaux')
aux = cd.SX.sym('aux')
# dt = cd.SX.sym('dt')

u = cd.vertcat(alphaux, aux)

zdot = cd.vertcat(vx*cd.cos(phi), vx*cd.sin(phi), (vx/inter_axle)*cd.tan(delta), alphaux, aux)

# xc = cd.SX.sym('xc', order + 1, 1)
# yc = cd.SX.sym('yc', order + 1, 1)
# contour_cost = gen_cost_func(order)
# L = contour_cost(pos=cd.vertcat(x, y), a=aux, alpha=alphaux, dt=dt, t=theta, t_dest=1.0, cx=xc, cy=yc)['cost']

L = (x-xt)**2 + (y-yt)**2 + alphaux**2 + aux**2

# Continuous time dynamics
f = cd.Function('f', [z, u], [zdot, L])

# Control discretization
N = 40 # number of control intervals
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
Xk = cd.SX.sym('X0', 5)
w.append(Xk)
lbw.append([-.3, 0, 0, 0, 0])
ubw.append([-.3, 0, 0, 0, 0])
w0.append([-.3, 0, 0, 0, 0])
x_plot.append(Xk)

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = cd.SX.sym('U_' + str(k), 2)
    w.append(Uk)
    lbw.append([-2*cd.pi, -1])
    ubw.append([ 2*cd.pi,  1])
    w0.append([0, 0])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []

    for j in range(d):
        Xkj = cd.SX.sym('X_'+str(k)+'_'+str(j), 5)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-cd.inf, -cd.inf, -cd.inf, -cd.pi/4,  0])
        ubw.append([ cd.inf,  cd.inf,  cd.inf,  cd.pi/4,  2])
        w0.append([0, 0, 0, 0, 0])

    # Loop over collocation points
    Xk_end = D[0]*Xk
    for j in range(1,d+1):
        # Expression for the state derivative at the collocation point
        xp = C[0,j]*Xk
        for r in range(d): xp = xp + C[r+1,j]*Xc[r]

        # Append collocation equations
        fj, qj = f(Xc[j-1], Uk)
        g.append(h*fj - xp)
        lbg.append([0, 0, 0, 0, 0])
        ubg.append([0, 0, 0, 0, 0])

        # Add contribution to the end state
        Xk_end = Xk_end + D[j]*Xc[j-1];

        # Add contribution to quadrature function
        J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = cd.SX.sym('X_' + str(k+1), 5)
    w.append(Xk)
    lbw.append([-cd.inf, -cd.inf, -cd.inf, -cd.pi/4,  0])
    ubw.append([ cd.inf,  cd.inf,  cd.inf,  cd.pi/4,  2])
    w0.append([0, 0, 0, 0, 0])
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end-Xk)
    lbg.append([0, 0, 0, 0, 0])
    ubg.append([0, 0, 0, 0, 0])

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
prob = {'f': J, 'x': w, 'g': g} # 'p': cd.vertcat(xt, yt, xc, yc)
solver = cd.nlpsol('solver', 'ipopt', prob);

# Function to get x and u trajectories from w
trajectories = cd.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full() # to numpy array
u_opt = u_opt.full() # to numpy array

# Plot the result
tgrid = np.linspace(0, T, N+1)
fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))
# plt.clf()
x_diff = [xt - x for x in x_opt[0]]
y_diff = [yt - y for y in x_opt[1]]
ax1.plot(tgrid, x_diff, '-', color='gray')
ax1.plot(tgrid, y_diff, '-', color='black')
ax1.step(tgrid, np.append(np.nan, u_opt[0]), '-.', color='green')
ax1.step(tgrid, np.append(np.nan, u_opt[1]), '-.', color='blue')
plt.xlabel('t')
ax1.legend(['xt - x','yt - y', 'a_x^u', 'alpha_x^u'])
ax1.grid()
ax1.set_ylim([-5, 5])

ax2.plot(x_opt[0], x_opt[1], '-', color='green', alpha=0.4)
ax2.plot([x_opt[0][0]], [x_opt[1][0]], marker='o', color='blue')
ax2.plot([xt], [yt], marker='x', color='blue')

ax2.set_ylim([-5, 5])
ax2.set_xlim([-5, 5])
ax2.set_xlabel('x')
ax2.grid()
plt.show()