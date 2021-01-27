import matplotlib.pyplot as plt 
import numpy as np

# Ant-v2_Test.npy
file_name = 'Ant-v2_Test.npy' # 'LunarLanderContinuous_Test.npy'
data = np.load(f"./results/{file_name}")
plt.plot(data)
plt.show()