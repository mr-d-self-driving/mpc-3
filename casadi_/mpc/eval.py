import os
import pandas as pd

import casadi_.mpc.config as cfg

direc = os.path.join(cfg.out_path, 'time')
for filename in os.listdir(direc):
    f = os.path.join(direc, filename)
    if os.path.isfile(f):
        csv_tmp = pd.read_csv(f)
        print(filename)
        ipopt_mean = csv_tmp['IN_IPOPT'].mean()
        nlp_mean = csv_tmp['IN_NLP'].mean()
        print('ipopt mean', f'{ipopt_mean:.4f}')
        print('nlp mean', f'{nlp_mean:.4f}')