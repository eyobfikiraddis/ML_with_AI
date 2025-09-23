import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

dot_ab = np.dot(a,b)
print(dot_ab)

norm_a = np.sqrt(np.dot(a,a))

A = np.array([[1,2],[3,4]])
A2 = A @ A
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)
print('dot', dot_ab, 'norm', norm_a)
print('A2\n', A2)
print('eigvals', eigvals)