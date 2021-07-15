import casadi as cd

def poly_deriv(coefs, order):
    coef_d = cd.SX.sym('c_deriv', order, 1)
    for i in range(coef_d.shape[0]):
        coef_d[i] = (order - i) * coefs[i]
    return coef_d

def gen_cost_func(order):

    a = cd.SX.sym('a')
    alpha = cd.SX.sym('alpha')
    dt = cd.SX.sym('dt')

    npoly = order + 1

    coefs_x = cd.SX.sym('cx', npoly, 1)
    coefs_y = cd.SX.sym('cy', npoly, 1)
    pos = cd.SX.sym('pos', 2, 1)

    t = cd.SX.sym('t')
    t_dest = cd.SX.sym('t_dest')
    
    deriv_coefs_x = poly_deriv(coefs_x, order)
    deriv_coefs_y = poly_deriv(coefs_y, order)

    path_poly_eval = (lambda cx, cy, d: cd.vertcat(cd.polyval(cx, d), cd.polyval(cy, d)))

    s_theta = path_poly_eval(coefs_x, coefs_y, t)
    r_pqs = s_theta - pos
    s_prime = path_poly_eval(deriv_coefs_x, deriv_coefs_y, t)
    n = s_prime / cd.norm_2(s_prime)
    res_proj = cd.mtimes([cd.transpose(r_pqs), n])
    e_l = cd.norm_2(res_proj) ** 2
    e_c = cd.norm_2(r_pqs - cd.mtimes([res_proj, n])) ** 2
    
    inputs = [pos, a, alpha, dt, t, t_dest, coefs_x, coefs_y]
    labels = ['pos', 'a', 'alpha', 'dt', 't', 't_dest', 'cx', 'cy']
    outputs = [5.*e_c + 1.*e_l + .5*(t - t_dest)**2 + 1.*(a**2) + 1.*(alpha**2) + 1.*(dt**2)] # [5.*e_c + 1.*e_l + .3*(t - t_dest)**2 + 1.*(a**2) + 1.*(alpha**2) + 1.*(dt**2)]

    control_cost = cd.Function('state_costs', inputs, outputs, labels, ['cost'])
    return control_cost
