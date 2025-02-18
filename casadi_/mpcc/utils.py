import casadi as cd
import pandas as pd
import numpy as np
import re

def get_timing(txt):
    pattern = r'Total CPU secs in IPOPT \(w/o function evaluations\)   =      (.*?) Total CPU secs in NLP function evaluations           =      (.*?)  EXIT'

    time = re.findall(pattern, txt)
    time = [tuple(float(v) for v in t) for t in time]
    return time

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

def compute_step(init, ts, D): # init = [x, y, phi, delta, vx, theta, alphaux, aux, dt]
    x, y, phi, delta, v, theta, alpha, a, dt = init
    
    x_ts = x + v*cd.cos(phi)*ts
    y_ts = y + v*cd.sin(phi)*ts
    phi_ts = phi + (v/D)*cd.tan(delta)*ts
    delta_ts = delta + alpha*ts
    v_ts = v + a*ts
    theta_ts = theta + v*dt*ts

    return [x_ts, y_ts, phi_ts, delta_ts, v_ts, theta_ts]

def compute_cost_step(init, cost_func, xc, yc, ts):
    x, y, phi, delta, v, theta, alpha, a, dt = init
    cost = cost_func(pos=cd.vertcat(x, y), a=a, alpha=alpha, dt=dt, t=theta, t_dest=1.0, cx=xc, cy=yc)['cost']*ts
    return cost

def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z

def prep_df(fn1, fn2):
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)

    df2 = df2.reindex(index=df2.index[::-1])
    df2['cost'] = df2.cost.cumsum()
    df2 = df2.reindex(index=df2.index[::-1])

    # scale time
    maxt_df1 = df1['time'].max()
    maxt_df2 = df2['time'].max()

    df1['time'] = df1['time'].div(maxt_df1)
    df2['time'] = df2['time'].div(maxt_df2)

    return df1, df2

def interpolate(df1, df2, cn='time'):
    column_names = list(df2.columns)
    df2_tmp = df2.values.tolist()
    for time in df1[cn].values:
        if time not in df2[cn].values:
            df2_tmp.append([time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    df2_fin = pd.DataFrame(df2_tmp, columns=column_names).sort_values(cn)
    df2_fin = df2_fin.reset_index(drop=True)
    df2_fin.index = df2_fin[cn]
    del df2_fin[cn]
    df2_fin = df2_fin.interpolate()
    df2_fin.reset_index(level=0, inplace=True)
    return df2_fin

def compare_costs(df1, df2, cn='time'):
    column_names = [cn, 'pred_cost', 'true_cost']
    data = []
    for i, t in enumerate(df1[cn].values):
        df1_cost = df1.iloc[i]['cost']
        tmp = df2.time[df2.time == t].index.tolist()
        df2_cost = df2.iloc[tmp[0]]['cost']
        data.append([t, df1_cost, df2_cost])

    return pd.DataFrame(data, columns=column_names)