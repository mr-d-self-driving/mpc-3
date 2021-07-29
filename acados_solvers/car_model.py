import casadi as cd
from acados_template import AcadosModel

def car_model():

    # constants
    D = 0.5

    model_name = "car_kinematic"

    x1 = cd.SX.sym('x1')
    y1 = cd.SX.sym('y1')
    phi = cd.SX.sym('phi')
    delta = cd.SX.sym('delta')
    v1 = cd.SX.sym('v1')
    x = cd.vertcat(x1, y1, phi, delta, v1)

    # controls
    alphaux = cd.SX.sym('alphaux')
    aux = cd.SX.sym('aux')
    u = cd.vertcat(aux, alphaux)

    x1_dot = cd.SX.sym('x1_dot')
    y1_dot = cd.SX.sym('y1_dot')
    phi_dot = cd.SX.sym('phi_dot')
    delta_dot = cd.SX.sym('delta_dot')
    v1_dot = cd.SX.sym('v1_dot')
    xdot = cd.vertcat(x1_dot, y1_dot, phi_dot, delta_dot, v1_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []

    # dynamics
    f_expl = cd.vertcat(v1*cd.cos(phi),
                        v1*cd.sin(phi),
                        (v1/D)*cd.tanh(delta),
                        alphaux,
                        aux)

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

def car_model_rk4(DT):
    model = car_model()

    x = model.x
    u = model.u
    nx = x.size()[0]

    ode = cd.Function('ode', [x, u], [model.f_expl_expr])

    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+DT/2*k1,u)
    k3 = ode(x+DT/2*k2,u)
    k4 = ode(x+DT*k3,  u)
    xf = x + DT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    print("built RK4 for pendulum model with DT = ", DT)
    print(xf)
    return model