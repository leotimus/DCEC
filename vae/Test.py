import numpy as np

a = np.array([[1, 2], [3, 4], [3, 4]])
b = np.array([5, 6, 7])

print(a.shape)
print(b.shape)

print(np.append(a, b, axis=0))