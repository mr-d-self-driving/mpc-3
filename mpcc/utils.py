import casadi as cd

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

    # print('\n')
    # print(theta, dt)
    
    x_ts = x + v*cd.cos(phi)*ts
    y_ts = y + v*cd.sin(phi)*ts
    phi_ts = phi + (v/D)*cd.tan(delta)*ts
    delta_ts = delta + alpha*ts
    v_ts = v + a*ts
    theta_ts = theta + v*dt*ts

    # print('\n', x, y, v, a, '\n')

    return [x_ts, y_ts, phi_ts, delta_ts, v_ts, theta_ts]

def merge_dict(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z