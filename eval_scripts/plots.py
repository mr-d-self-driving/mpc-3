import os
import pandas as pd
import matplotlib.pyplot as plt

curr_path = os.path.dirname(__file__) # sorta hacky

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10, 5))
col_i = 0
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
legend = []

direc = os.path.join(curr_path, 'mpcc_data')
for filename in os.listdir(direc):
    f = os.path.join(direc, filename)
    if os.path.isfile(f):
        csv_tmp = pd.read_csv(f, header=None)[0].to_numpy()
        csv_tmp = csv_tmp[2:-2]

        lgd_tmp = filename.split('_')
        library = lgd_tmp[0]
        ln_style = 'solid'
        if library == 'CasADi':
            mark_style = '.' if 'rk4' in filename else 's'

            method = lgd_tmp[1]
            solver = 'DMS' if 'rk4' in filename else 'DC'
            legend.append(library + ' ' + method + ' ' + solver + ' ({:.3f})'.format(csv_tmp.mean()))
        else:
            mark_style = '^'

            method = lgd_tmp[1]
            legend.append(library + ' ' + method.split('.')[0] + ' ({:.3f})'.format(csv_tmp.mean()))

        if mark_style == 's' or mark_style == '^':
            ax.plot(range(csv_tmp.shape[0]), csv_tmp, marker=mark_style, linestyle=ln_style, markersize=5, color=colors[col_i], alpha=0.3)
        else:
            ax.plot(range(csv_tmp.shape[0]), csv_tmp, marker=mark_style, linestyle=ln_style, color=colors[col_i], alpha=0.3)
        col_i += 1

        # csv_tmp = pd.read_csv(f, header=None)
        # print(filename)
        # time_simple_mean = csv_tmp[0].to_numpy()[5:-5].mean()
        # print('time_simple mean', f'{time_simple_mean:.4f}')
        # ipopt_mean = csv_tmp['IN_IPOPT'].mean()
        # nlp_mean = csv_tmp['IN_NLP'].mean()
        # print('ipopt mean', f'{ipopt_mean:.4f}')
        # print('nlp mean', f'{nlp_mean:.4f}')

ax.set_title('MPCC Timing Stats (T=10 N=40)')
ax.set_ylabel('Seconds')
ax.set_xlabel('Iteration Number')
# ax.set_ylim([0.0, 0.08])
plt.legend(legend)
plt.grid(True)
plt.savefig(os.path.join(curr_path, 'mpcc_simple_time_stats.png'), dpi=300)