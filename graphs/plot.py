import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
df = pd.read_csv('progress.csv',header=0)
column_names = df.columns
progress_mat = np.array(df.values)



y_data = progress_mat[:,0]
x_data = range(y_data.shape[0])

_, ax = plt.subplots()

ax.plot(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Fit')
# Shade the confidence interval
#ax.fill_between(x_data, y_data - np.random.rand(100), y_data + np.random.rand(100), color = '#539caf', alpha = 0.4, label = '95% CI')
# Label the axes and provide a title
ax.set_title(r'$\alpha > \beta$')
ax.set_xlabel('x_label')
ax.set_ylabel('y_label')

# Display legend
ax.legend(loc = 'best')
plt.show()

