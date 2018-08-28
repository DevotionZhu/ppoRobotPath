import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
df = pd.read_csv('./graphs-10cpu/progress.csv',header=0)
column_names_10cpu = df.columns.tolist()
progress_10cpu = np.array(df.values)

df = pd.read_csv('./graphs-20cpu/progress.csv',header=0)
column_names_20cpu = df.columns.tolist()
progress_20cpu = np.array(df.values)

df = pd.read_csv('./graphs-25cpu/progress.csv',header=0)
column_names_25cpu = df.columns.tolist()
progress_25cpu = np.array(df.values)

df = pd.read_csv('./graphs-30cpu/progress.csv',header=0)
column_names_30cpu = df.columns.tolist()
progress_30cpu = np.array(df.values)

df = pd.read_csv('./graphs-40cpu/progress.csv',header=0)
column_names_40cpu = df.columns.tolist()
progress_40cpu = np.array(df.values)

max_iter = 100
x_data = range(max_iter)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.3)

j=0
figure_titles=['$S[\pi]$','$L^{VF}$','$L^{CLIP}$','$- mean(R_0)$']
title_index_10cpu = [column_names_10cpu.index('loss_ent'),column_names_10cpu.index('loss_vf_loss'),column_names_10cpu.index('loss_pol_surr'),column_names_10cpu.index('EpRewMean')]
title_index_20cpu = [column_names_20cpu.index('loss_ent'),column_names_20cpu.index('loss_vf_loss'),column_names_20cpu.index('loss_pol_surr'),column_names_20cpu.index('EpRewMean')]
title_index_25cpu = [column_names_25cpu.index('loss_ent'),column_names_25cpu.index('loss_vf_loss'),column_names_25cpu.index('loss_pol_surr'),column_names_25cpu.index('EpRewMean')]
title_index_30cpu = [column_names_30cpu.index('loss_ent'),column_names_30cpu.index('loss_vf_loss'),column_names_30cpu.index('loss_pol_surr'),column_names_30cpu.index('EpRewMean')]
title_index_40cpu = [column_names_40cpu.index('loss_ent'),column_names_40cpu.index('loss_vf_loss'),column_names_40cpu.index('loss_pol_surr'),column_names_40cpu.index('EpRewMean')]


for i in range(4):
    y_data_10cpu = progress_10cpu[:max_iter, title_index_10cpu[i]]
    y_data_20cpu = progress_20cpu[:max_iter, title_index_20cpu[i]]
    y_data_25cpu = progress_25cpu[:max_iter, title_index_25cpu[i]]
    y_data_30cpu = progress_30cpu[:max_iter, title_index_30cpu[i]]
    y_data_40cpu = progress_40cpu[:max_iter, title_index_40cpu[i]]
    
    plt.subplot(2, 2, j+1)
    figure_title = figure_titles[j]
    if i == 1 or i == 30:
        plt.semilogy(x_data, y_data_10cpu, lw = 1, label = '10 cpu')
        plt.semilogy(x_data, y_data_20cpu, lw = 1, label = '20 cpu')
        plt.semilogy(x_data, y_data_25cpu, lw = 1, label = '25 cpu')
        plt.semilogy(x_data, y_data_30cpu, lw = 1, label = '30 cpu')
        plt.semilogy(x_data, y_data_40cpu, lw = 1, label = '40 cpu')
    elif i==3:
        plt.semilogy(x_data, -y_data_10cpu, lw = 1, label = '10 cpu')
        plt.semilogy(x_data, -y_data_20cpu, lw = 1, label = '20 cpu')
        plt.semilogy(x_data, -y_data_25cpu, lw = 1, label = '25 cpu')
        plt.semilogy(x_data, -y_data_30cpu, lw = 1, label = '30 cpu')
        plt.semilogy(x_data, -y_data_40cpu, lw = 1, label = '40 cpu')
    else:
        plt.plot(x_data, y_data_10cpu, lw = 1, label = '10 cpu')
        plt.plot(x_data, y_data_20cpu, lw = 1, label = '20 cpu')
        plt.plot(x_data, y_data_25cpu, lw = 1, label = '25 cpu')
        plt.plot(x_data, y_data_30cpu, lw = 1, label = '30 cpu')
        plt.plot(x_data, y_data_40cpu, lw = 1, label = '40 cpu')
        
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

