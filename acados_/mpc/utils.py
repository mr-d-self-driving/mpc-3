import numpy as np

def compute_step(init, ts, D): # init = [x, y, phi, delta, vx, aux, alphaux]
    x, y, phi, delta, v, a, alpha = init
    
    x_ts = x + v*np.cos(phi)*ts
    y_ts = y + v*np.sin(phi)*ts
    phi_ts = phi + (v/D)*np.tan(delta)*ts
    delta_ts = delta + alpha*ts
    v_ts = v + a*ts

    return np.array([x_ts, y_ts, phi_ts, delta_ts, v_ts])