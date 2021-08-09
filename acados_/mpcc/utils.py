import casadi as cd
import pandas as pd
import numpy as np

def gen_t(pts1, pts2):
    tpts = [0]
    for i, pt in enumerate(pts1):
        if i != 0:
            dist_tmp = (pts1[i] - pts1[i-1]) ** 2 + (pts2[i] - pts2[i-1]) ** 2
            tpts += [cd.sqrt(dist_tmp) + tpts[-1]]
    maxt = tpts[-1]
    tpts = [t/maxt for t in tpts]
    return tpts

def get_curve(curve, prev=None):
    xpts, ypts = curve['xpts'], curve['ypts']
    order = curve['order']

    if prev is not None:
        init_ts = prev
    else:
        init_ts = curve['init_ts']
    
    xs, ys = xpts[0], ypts[0]
    xf, yf = xpts[-1], ypts[-1]

    tpts = gen_t(xpts, ypts)
    xpoly = np.polynomial.polynomial.Polynomial.fit(tpts, xpts, order)
    ypoly = np.polynomial.polynomial.Polynomial.fit(tpts, ypts, order)
    cx = list(xpoly)[::-1]
    cy = list(ypoly)[::-1]
    
    return xs, ys, xf, yf, init_ts, xpts, ypts, tpts, xpoly, ypoly, cx, cy, order


def compute_step(init, ts, D): # init = [x, y, phi, delta, vx, theta, aux, alphaux, dt]
    x, y, phi, delta, v, theta, a, alpha, dt = init
    
    x_ts = x + v*np.cos(phi)*ts
    y_ts = y + v*np.sin(phi)*ts
    phi_ts = phi + (v/D)*np.tan(delta)*ts
    delta_ts = delta + alpha*ts
    v_ts = v + a*ts
    theta_ts = theta + v*dt*ts

    return np.array([x_ts, y_ts, phi_ts, delta_ts, v_ts, theta_ts])