from acados_template import AcadosOcp
from acados_.mpc.model import car_model
import numpy as np
import scipy.linalg

def build_ocp(init_ts, target, Tf, N, D, export_dir):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = car_model(D)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    
    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))

    # set dimensions
    ocp.dims.N = N

    # set cost
    Q = np.diag([1e0, 1e0, 1e-4, 1e-4, 1e-4])
    R = np.diag([1e0, 1e0])

    unscale = N / Tf

    ocp.cost.W_e = Q / unscale
    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[5,0] = 1.0
    Vu[6,1] = 1.0
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref  = np.array(target)
    ocp.cost.yref_e = np.array(target[:-2])

    # set constraints
    deltamax = np.pi/4
    ocp.constraints.lbx = np.array([-deltamax, 0])
    ocp.constraints.ubx = np.array([+deltamax, 2])
    ocp.constraints.idxbx = np.array([3, 4])

    amax = 1.0
    alphamax = np.pi
    ocp.constraints.lbu = np.array([-amax, -alphamax])
    ocp.constraints.ubu = np.array([+amax, +alphamax])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.x0 = init_ts

    ocp.solver_options.nlp_solver_max_iter = 400

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp.code_export_directory = export_dir

    return ocp, simX, simU