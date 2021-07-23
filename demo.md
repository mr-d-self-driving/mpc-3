# Model Predictive Control (MPC)

## Toy unicycle MPC
### System
![uni_system](/img/eqs/uni_system.svg)
### Controls
![uni_controls](/img/eqs/uni_controls.svg)
### Demo
![mpc](/demos/mpc/toy_uni.gif)

## Car contouring control
### System
![car_system](/img/eqs/car_system.svg)
### Controls
![car_controls](/img/eqs/car_controls.svg)
### 3rd-degree polynomial RK4
![3_mpcc_rk4](/demos/mpcc_rk4/3deg_time.gif)
### 3rd-degree polynomial Direct Collocation
![3_mpcc_colloc](/demos/mpcc_colloc/3deg_time.gif)
### 5th-degree polynomial RK4
![5_mpcc_rl4](/demos/mpcc_rk4/5deg_time.gif)

## Hessian sparsity
### RK4
![rk4_s](/img/sparsity_im/sparsity_rk4.png)
### Direct collocation
![colloc_s](/img/sparsity_im/sparsity_colloc.png)