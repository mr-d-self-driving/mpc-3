import casadi as cd
from casadi.casadi import PrintableCommon_swigregister
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

def interpolate(df1, df2):
    column_names = list(df2.columns)
    df2_tmp = df2.values.tolist()
    for time in df1['time'].values:
        if time not in df2['time'].values:
            df2_tmp.append([time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    df2_fin = pd.DataFrame(df2_tmp, columns=column_names).sort_values('time')
    df2_fin = df2_fin.reset_index(drop=True)
    df2_fin = df2_fin.interpolate()
    return df2_fin

def compare_costs(df1, df2):
    column_names = ['time', 'pred_cost', 'true_cost']
    data = []
    for i, t in enumerate(df1['time'].values):
        df1_cost = df1.iloc[i]['cost']
        tmp = df2.time[df2.time == t].index.tolist()
        df2_cost = df2.iloc[tmp[0]]['cost']
        data.append([t, df1_cost, df2_cost])

    return pd.DataFrame(data, columns=column_names)