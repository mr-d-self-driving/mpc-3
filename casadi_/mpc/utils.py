import casadi as cd
import re

def compute_step(init, ts, D): # init = [x, y, phi, delta, vx, alphaux, aux]
    x, y, phi, delta, v, alpha, a = init
    
    x_ts = x + v*cd.cos(phi)*ts
    y_ts = y + v*cd.sin(phi)*ts
    phi_ts = phi + (v/D)*cd.tan(delta)*ts
    delta_ts = delta + alpha*ts
    v_ts = v + a*ts

    return [x_ts, y_ts, phi_ts, delta_ts, v_ts]

def merge_dict(x, y):
    z = x.copy()
    z.update(y)
    return z

def get_timing(txt):
    pattern = r'Total CPU secs in IPOPT \(w/o function evaluations\)   =      (.*?) Total CPU secs in NLP function evaluations           =      (.*?)  EXIT'

    time = re.findall(pattern, txt)
    time = [tuple(float(v) for v in t) for t in time]
    return time