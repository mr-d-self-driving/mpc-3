import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpc import solve

init_vals = [0, 1, 0, 0, 0] # [x, y, theta, v, omega]
init_x, init_y = init_vals[0], init_vals[1]
fin_x, fin_y = -0.75, -1.0

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(10, 5))
unicycle, = ax2.plot([init_x], [init_y], marker='o', color='blue')
target, = ax2.plot([fin_x], [fin_y], marker='x', color='blue')

def init_traj_plot():
    x_opt, y_opt, theta_opt, v_opt, omega_opt, a_opt, alpha_opt = solve(init_x, init_y, init_vals[2], init_vals[3], init_vals[4], fin_x, fin_y)
