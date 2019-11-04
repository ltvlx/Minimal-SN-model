import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os


np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:.2f}'.format
np.random.seed(2)


path = 'res-N=5-K=3-rd=zero_prop/'
path = 'res-N=5-K=3-rd=zero_pair/'
path = 'res-N=5-K=3-rd=mean/'

df_inv = pd.read_excel(path + 'history.xlsx', sheet_name = 'inventory', index_col=0)
df_del = pd.read_excel(path + 'history.xlsx', sheet_name = 'Deliveries', index_col=0)

del_x = []
del_y = []
for i, row in df_del.iterrows():
    if row['total'] >= 0.001:
        del_x.append(i)
        del_y.append(row['total'])



dsum = df_inv.sum(axis=1)
dmin = df_inv.max(axis=1)
dmax = df_inv.min(axis=1)
df_inv['sum'] = dsum
df_inv['max'] = dmin
df_inv['min'] = dmax


print(df_inv)


fig, ax = plt.subplots(2, 1, figsize=(10,5))

ax[0].plot([0, 1000], [0, 0], color='black', alpha = 0.6)
ax[0].plot(df_inv['sum'], label='total')
ax[0].plot(df_inv['max'], label='max')
ax[0].plot(df_inv['min'], label='min')
ax[0].legend()
ax[0].set_ylabel('inventory')
ax[0].set_xlim(0, 1000)
ax[0].grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')



# ax[1].plot(df_del['total'], color='C3', label='deliveries')
ax[1].scatter(del_x, del_y, s=10, facecolors='none', edgecolors='C3',alpha=1.0)
ax[1].set_ylabel('total\ndelivery')
ax[1].set_xlabel('t')
ax[1].set_xlim(0, 1000)
ax[1].set_ylim(-5, 80)
ax[1].grid(alpha = 0.4, linestyle = '--', linewidth = 0.2, color = 'black')



plt.savefig(path + "history_analysis.png", bbox_inches = 'tight', pad_inches=0.1, dpi=400)
# plt.show()
plt.close()



