import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
df = pd.read_csv('progress.csv',header=0)
column_names = df.columns.tolist()
progress_20cpu = np.array(df.values)

x_data = range(progress_20cpu.shape[0])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.2, hspace=0.5)

j=0
figure_titles=['$S[\pi]$','$L^{VF}$','$L^{CLIP}$','$mean(R_0)$']
title_index = [column_names.index('loss_ent'),column_names.index('loss_vf_loss'),column_names.index('loss_pol_surr'),column_names.index('EpRewMean')]
for i in title_index:
    y_data_20cpu = progress_20cpu[:,i]
    plt.subplot(2, 2, j+1)
    figure_title = figure_titles[j]
    plt.plot(x_data, y_data_20cpu, lw = 1, label = '20 cpu')
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

