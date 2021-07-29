# Model Predictive Control (MPC)

Implemented models:
- Car kinematic MPC w/ direct multiple shooting or direct collocation
  - Use solvers `mpc_colloc.py` or `mpc_rk4.py`
- Car kinematic MP Contour Control (MPCC) w/ direct multiple shooting or direct collocation
  -  Use solvers `mpcc_colloc.py` or `mpcc_rk4.py`
- Unicycle kinematic (only mpc w/ direct multiple shooting available)

## Running demos
Two options are available for demos, either plotting a single trajectory computed by the MPC(C) or displaying an animation of the entire path from the starting location to the target location. Some configurable options for the default models (i.e. time horizon, number of control intervals, etc.) are available in `mpc(c)/config.py`. Demos are run via `run.py`.

## Editing models
By default, all solvers in `solvers/` use the car kinematic model. For example, `mpcc_[method].py` uses the following system of equations and controls:

![car_system](/img/eqs/car_system.svg)

![car_controls](/img/eqs/car_controls.svg)

Which looks like this in the code:
```python
## System Variables
x = cd.SX.sym('x')
y = cd.SX.sym('y')
phi = cd.SX.sym('phi')
delta = cd.SX.sym('delta')
vx = cd.SX.sym('vx')
theta = cd.SX.sym('theta')

z = cd.vertcat(x, y, phi, delta, vx, theta)

## Control variables
alphaux = cd.SX.sym('alphaux')
aux = cd.SX.sym('aux')
dt = cd.SX.sym('dt')

u = cd.vertcat(alphaux, aux, dt)

zdot = cd.vertcat(vx*cd.cos(phi), vx*cd.sin(phi), (vx/inter_axle)*cd.tan(delta), alphaux, aux, vx*dt)
```
Therefore, using a different model requires declaring all the relevant system variables `s1 = cd.SX.sym('s1'); s2 = cd.SX.sym('s2'); ...`, control variables `c1 = cd.SX.sym('c1'); c2 = cd.SX.sym('c2'); ...` and them combining them into the system vector `z = cd.vertcat(s1, s1, ...)`, control vector `u = cd.vertcat(c1, c2, ...)`. Additionally, change the `zdot` vector according to the new system constraints. Make sure to also edit the bounds (`lbw` & `ubw`) in the NLP formulation.

## Installing HSL solvers
See `hsl.md` for instructions if CasADI is installed via pip.