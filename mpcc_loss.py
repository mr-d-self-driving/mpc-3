from casadi import *

def poly_deriv(coefs, order):
    coef_d = SX.sym('c_deriv', order, 1)
    
    for i in range(coef_d.shape[0]):
        coef_d[i] = (order - i) * coefs[i]

    print('\n\n', coef_d, '\n\n')
    return coef_d
    
    # print('\n\n', vcat(coef_d), '\n\n')

    # coef_d = [(order - i) * coefs[i] for i in range(order)]
    # return vcat(coef_d)

def gen_cost_func(order):

    npoly = order + 1

    pos = SX.sym('pos', 2, 1)

    coefs_x = SX.sym('cx', npoly, 1)
    coefs_y = SX.sym('cy', npoly, 1)

    t = SX.sym('t')
    t_dest = SX.sym('t_dest')
    
    deriv_coefs_x = poly_deriv(coefs_x, order)
    deriv_coefs_y = poly_deriv(coefs_y, order)

    path_poly_eval = (lambda cx, cy, d: vertcat(polyval(cx, t), polyval(cy, t)))

    s_theta = path_poly_eval(coefs_x, coefs_y, t)
    s_prime = path_poly_eval(deriv_coefs_x, deriv_coefs_y, t)

    r_pqs = s_theta - pos
    n = s_prime / norm_2(s_prime)
    res_proj = mtimes([transpose(r_pqs), n])
    e_l = norm_2(res_proj) ** 2
    e_c = norm_2(r_pqs - mtimes([res_proj, n])) ** 2
    
    inputs = [pos, t, t_dest, coefs_x, coefs_y]
    labels = ['pos', 't', 't_dest', 'cx', 'cy']
    outputs = [e_c + e_l +(t - t_dest)**2]

    control_cost = Function('state_costs', inputs, outputs, labels, ['cost'])
    return control_cost
