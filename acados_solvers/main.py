import sys
sys.path.insert(0, '../common')

from acados_template import AcadosOcp, AcadosOcpSolver
from car_model import car_model
import numpy as np
import scipy.linalg
from utils import plot_car

# create ocp object to formulate the OCP
ocp = AcadosOcp()

# set model
model = car_model()
ocp.model = model

Tf = 10.0 # Time horizon
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = 40 # Control intervals

# set dimensions
ocp.dims.N = N

# set cost
Q = np.diag([1e2, 1e2, 1., 1., 1.])
R = np.diag([1., 1.])

ocp.cost.W_e = Q
ocp.cost.W = scipy.linalg.block_diag(Q, R)

ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx)

Vu = np.zeros((ny, nu))
Vu[5,0] = 1.0
Vu[6,1] = 1.0
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.eye(nx)

ocp.cost.yref  = np.zeros((ny, ))
ocp.cost.yref_e = np.zeros((ny_e, ))

# set constraints
xmax = 100
ymax = 100
phimax = 100
deltamax = np.pi/4
ocp.constraints.lbx = np.array([-xmax, -ymax, -phimax, -deltamax, 0])
ocp.constraints.ubx = np.array([+xmax, +ymax, +phimax, +deltamax, 2])
ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4])

alphamax = 2*np.pi
amax = 1.0
ocp.constraints.lbu = np.array([-amax, -alphamax])
ocp.constraints.ubu = np.array([+amax, +alphamax])
ocp.constraints.idxbu = np.array([0, 1])

ocp.constraints.x0 = np.array([1.0, -1.0, -np.pi, 0.0, 0.0])

# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.print_level = 1
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

# set prediction horizon
ocp.solver_options.tf = Tf

ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

simX = np.ndarray((N+1, nx))
simU = np.ndarray((N, nu))

status = ocp_solver.solve()

if status != 0:
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
    raise Exception('acados returned status {}. Exiting.'.format(status))

# get solution
for i in range(N):
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
print(ocp_solver.get_cost())
print(ocp_solver.get_stats('time_qp'))
simX[N,:] = ocp_solver.get(N, "x")

ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

plot_car(np.linspace(0, Tf, N+1), [amax, alphamax], simU, simX, latexify=False)