from acados_template import AcadosOcp
from acados_.mpcc.model import car_model
import numpy as np

def build_ocp(init_ts, order, Tf, N, D, export_dir):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = car_model(D, order)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    np_ = model.p.size()[0]

    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))

    # set dimensions
    ocp.dims.N  = N

    # set cost
    ocp.cost.cost_type = 'EXTERNAL'

    # set constraints
    deltamax = np.pi/4
    ocp.constraints.lbx = np.array([-deltamax, 0, 0])
    ocp.constraints.ubx = np.array([+deltamax, 2, 1])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    amax = 1.0
    alphamax = 2*np.pi
    ocp.constraints.lbu = np.array([-amax, -alphamax, 0])
    ocp.constraints.ubu = np.array([+amax, +alphamax, 1])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.x0 = init_ts

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

    ocp.parameter_values = np.zeros(np_)

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp.code_export_directory = export_dir

    return ocp, simX, simU