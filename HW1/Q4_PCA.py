import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

data = np.array([[82,87,94,101,107,114,115,112,112,103,94,83],
                [77,81,87,94,104,109,110,108,107,97,89,81],
                [76,73,72,82,88,92,95,97,93,88,78,75],
                [78,83,91,100,107,110,113,112,111,102,90,77],
                [93,92,98,105,104,109,109,105,111,111,101,92],
                [95,95,98,106,102,112,108,105,113,108,100,92],
                [80,83,88,96,106,117,118,115,116,106,89,78],
                [76,78,88,95,105,115,114,112,109,104,87,74],
                [88,90,93,98,97,101,99,98,111,107,100,88],
                [79,81,87,94,97,103,99,98,106,102,86,76],
                [86,87,94,96,101,109,109,105,104,103,97,89],
                [89,89,95,103,105,110,104,104,106,108,96,90]], dtype=np.float64)
n, m = data.shape

# a
def standardize(data):
    data_n = data
    mean = []
    deviation = []
    for i in range(n):
        sum = 0
        mean.append(np.mean(data[:, i]))
        for j in range(n):
            sum += (data[j][i] - mean[i]) ** 2
        deviation.append(sqrt(sum/(n-1)))
    for i in range(n):
        for j in range(n):
            data_n[j][i] = (data_n[j][i] - mean[i]) / deviation[i]
    return data_n

data_n = standardize(data)

print('Standardized matrix:')
a = pd.DataFrame(data_n)
print(a.round(2))

# covariance matrix
covariance = np.eye(n, dtype=np.float64)
for i in range(1, n):
    for j in range(i):
        sum = 0
        for k in range(n):
            sum += (data_n[k][i]) * (data_n[k][j])
        covariance[i][j] = sum / (n-1)
        covariance[j][i] = covariance[i][j]
print('Covariance matrix:')
a = pd.DataFrame(covariance)
print(a.round(2))

# b
def eigenvalue(A, v):
    Av = A @ v
    return Av.dot(v)

def powerIteration(data, n):
    np.random.seed(0)
    v = np.random.rand(n)
    v /=np.linalg.norm(v)
    ev = eigenvalue(data, v)
    while True:
        x_new = data.dot(v)
        v_new = x_new / np.linalg.norm(x_new)
        ev_new = eigenvalue(data, v_new)
        if np.abs(ev - ev_new) < 0.00001:
            break
        v = v_new
        ev = ev_new
    return ev_new, v_new

ev_vec = []
v_vec = []
_covariance = covariance.copy()
while len(ev_vec) < n:
    ev, v = powerIteration(_covariance, n)
    ev_vec.append(ev)
    v_vec.append(v)
    for i in range(n):
        _covariance[i] -= _covariance[i].dot(v) * v
print('Top 3 principle components: ')
a = pd.DataFrame(v_vec[:3])
print(a.T.round(2))
print('Cumulative percentage: ')
print('{}%'.format(np.sum(ev_vec[:3])*100 / np.sum(ev_vec)))

# u, s, vh = np.linalg.svd(covariance)
# print('Top 3 principle components: ')
# a = pd.DataFrame(s)
# print(a.round(2))
# print('Top 3 principle components: ')
# a = pd.DataFrame(vh)
# print(a.round(2))


# c
rotation_mat = np.zeros((3, 12), dtype=np.float64)
for i in range(3):
    rotation_mat[i] = v_vec[i]
recast_data = rotation_mat @ data.transpose()
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Recast data by PCA")
ax.set_xlabel("First Principle component")
ax.set_ylabel("Second Principle component")
ax.set_zlabel("Thrid Principle component")
ax.scatter(recast_data[0], recast_data[1], recast_data[2])
for i_x, i_y, i_z in zip(recast_data[0], recast_data[1], recast_data[2]):
    ax.text(i_x, i_y, i_z, '({}, {}, {})'.format(round(i_x, 2), round(i_y, 2), round(i_z, 2)))
plt.show()

print('Recast data')
a = pd.DataFrame(recast_data)
print(a.T.round(2))
