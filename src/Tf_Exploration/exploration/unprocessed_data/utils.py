import numpy as np

data = np.loadtxt('iris.data', delimiter=',', dtype=np.ndarray)
np.save('iris.npy', data)
