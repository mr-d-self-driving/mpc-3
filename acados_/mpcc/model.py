import casadi as cd
from acados_template import AcadosModel

def car_model(D):
    model_name = "car_kinematic"

    x1 = cd.SX.sym('x1')
    y1 = cd.SX.sym('y1')
    phi = cd.SX.sym('phi')
    delta = cd.SX.sym('delta')
    v1 = cd.SX.sym('v1')
    theta = cd.SX.sym('theta')
    x = cd.vertcat(x1, y1, phi, delta, v1, theta)

    # controls
    aux = cd.SX.sym('aux')
    alphaux = cd.SX.sym('alphaux')
    dt = cd.SX.syn('dt')
    u = cd.vertcat(aux, alphaux, dt)

    x1_dot = cd.SX.sym('x1_dot')
    y1_dot = cd.SX.sym('y1_dot')
    phi_dot = cd.SX.sym('phi_dot')
    delta_dot = cd.SX.sym('delta_dot')
    v1_dot = cd.SX.sym('v1_dot')
    theta_dot = cd.SX.sym('theta_dot')
    xdot = cd.vertcat(x1_dot, y1_dot, phi_dot, delta_dot, v1_dot, theta_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []

    # dynamics
    f_expl = cd.vertcat(v1*cd.cos(phi),
                        v1*cd.sin(phi),
                        (v1/D)*cd.tanh(delta),
                        alphaux,
                        aux,
                        v1*dt)

    f_impl = xdot - f_expl

    model = AcadosModel()

    # Define model struct
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    return model