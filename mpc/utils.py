import casadi as cd

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