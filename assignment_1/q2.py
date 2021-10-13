#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

fpath = os.path.join('data', 'steps.csv')
df = pd.read_csv(fpath)

data = df.iloc[:,[1,7]]

data = data - data.mean()

t = np.linspace(0, 0.01*len(data), len(data))
plt.plot(t[500:1000], data[500:1000])
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.legend(['std_acc_x', 'std_gyro_z'])

zero_crossings = np.round(np.mean(np.sum(np.diff(np.sign(data), axis = 0) > 0, axis = 0)))
print(f'Steps: {zero_crossings}')
# %%
