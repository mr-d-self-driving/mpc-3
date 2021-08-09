import os
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt

import casadi_.mpcc.config as cfg

plt.style.use('ggplot')

name = 'IN_NLP'
fig, ax = plt.subplots(figsize=(10, 5))
col_i = 0
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
legend = []

direc = os.path.join(cfg.out_path, 'time')
for filename in os.listdir(direc):
    f = os.path.join(direc, filename)
    if os.path.isfile(f) and 'compiled' in filename:
        csv_tmp = pd.read_csv(f)[name].to_numpy()
        ln_style = 'dashed' if 'ma75' in filename else 'solid'
        mark_style = '.' if 'rk4' in filename else 's'
        if mark_style == 's':
            ax.plot(range(csv_tmp.shape[0]), csv_tmp, marker=mark_style, linestyle=ln_style, markersize=5, color=colors[col_i], alpha=0.3)
        else:
            ax.plot(range(csv_tmp.shape[0]), csv_tmp, marker=mark_style, linestyle=ln_style, color=colors[col_i], alpha=0.3)
        col_i += 1

        lgd_tmp = filename.split('_')
        method = ''.join(v.upper() for v in lgd_tmp[1] if isinstance(v, str))
        solver = 'DMS' if 'rk4' in filename else 'DC'
        legend.append(method + ' ' + solver)

        # csv_tmp = pd.read_csv(f)
        # print(filename)
        # ipopt_mean = csv_tmp['IN_IPOPT'].mean()
        # nlp_mean = csv_tmp['IN_NLP'].mean()
        # print('ipopt mean', f'{ipopt_mean:.4f}')
        # print('nlp mean', f'{nlp_mean:.4f}')

ax.set_title('{} CPU Seconds'.format(name))
ax.set_ylabel('Seconds')
ax.set_xlabel('Iteration Number')
plt.legend(legend)
plt.grid(True)
plt.savefig(os.path.join(cfg.out_path, 'eval', 'mpc_nlp_time_stats.png'), dpi=300)