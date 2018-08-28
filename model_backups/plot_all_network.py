import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
df = pd.read_csv('./graphs-30cpu/progress.csv',header=0)
column_names_30cpu = df.columns.tolist()
progress_30cpu = np.array(df.values)

df = pd.read_csv('./graphs-30cpu-2h256/progress.csv',header=0)
column_names_30cpu_2h256 = df.columns.tolist()
progress_30cpu_2h256 = np.array(df.values)



max_iter = 300
x_data = range(max_iter)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)

j=0
figure_titles=['$S[\pi]$','$L^{VF}$','$L^{CLIP}$','$- mean(R_0)$']
title_index_30cpu = [column_names_30cpu.index('loss_ent'),column_names_30cpu.index('loss_vf_loss'),column_names_30cpu.index('loss_pol_surr'),column_names_30cpu.index('EpRewMean')]
title_index_30cpu_2h256 = [column_names_30cpu_2h256.index('loss_ent'),column_names_30cpu_2h256.index('loss_vf_loss'),column_names_30cpu_2h256.index('loss_pol_surr'),column_names_30cpu_2h256.index('EpRewMean')]


for i in range(4):
    y_data_30cpu = progress_30cpu[:max_iter, title_index_30cpu[i]]
    y_data_30cpu_2h256 = progress_30cpu_2h256[:max_iter, title_index_30cpu_2h256[i]]
    
    plt.subplot(2, 2, j+1)
    figure_title = figure_titles[j]
    if i == 1 or i == 30:
        plt.semilogy(x_data, y_data_30cpu, lw = 1, label = '30 cpu, 3 hidden 512')
        plt.semilogy(x_data, y_data_30cpu_2h256, lw = 1, label = '30 cpu, 2 hidden 256')
    elif i==3:
        plt.semilogy(x_data, -y_data_30cpu, lw = 1, label = '30 cpu, 3 hidden 512')
        plt.semilogy(x_data, -y_data_30cpu_2h256, lw = 1, label = '30 cpu, 2 hidden 256')
    else:
        plt.plot(x_data, y_data_30cpu, lw = 1, label = '30 cpu, 3 hidden 512')
        plt.plot(x_data, y_data_30cpu_2h256, lw = 1, label = '30 cpu, 2 hidden 256')
        
    # Shade the confidence interval
    #plt.fill_between(x_data, y_data - np.random.rand(100), y_data + np.random.rand(100), color = '#539caf', alpha = 0.4, label = '95% CI')
    # Label the pltes and provide a title
    plt.title(figure_title)
    plt.xlabel('iterations')
    plt.ylabel('')

    # Display legend
    plt.legend(loc = 'best')
    j += 1
    
plt.show()

