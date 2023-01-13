import numpy as np
import pandas as pd

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

# LU factorization
LU_data = data.copy()
L = np.eye(n, dtype=np.float64)
for column in range(n):
    for row in range(column + 1, n):
        L[row][column] = LU_data[row][column]/LU_data[column][column]
        LU_data[row] =  LU_data[row] - LU_data[column]*L[row][column]
print('L matrix:')
a = pd.DataFrame(L)
print(a.round(2))
print('U matrix:')
a = pd.DataFrame(LU_data)
print(a.round(2))
# print('Origin matrix:')
# a = pd.DataFrame(L @ LU_data)
# print(a.round(2))

# Gram-Schmidt
GS_data = data
Q = np.zeros((n,n), dtype=np.float64)
R = np.zeros((n,n), dtype=np.float64)
for i in range(0, n):
    Q[:, i] = GS_data[:,i]
    for j in range(i):
        R[j][i] = Q[:, j].dot(GS_data[:, i])
        Q[:, i] -= R[j][i]*Q[:, j]
    R[i][i] = np.linalg.norm(Q[:, i])
    Q[:, i] /= R[i][i]

R_inv = np.zeros((n,n), dtype=np.float64)
for j in range(n):
    for i in range(j):
        for k in range(j):
            R_inv[i][j] = R_inv[i][j] + R_inv[i][k] * R[k][j]
    for k in range(j):
        R_inv[k][j] = -R_inv[k][j] / R[j][j]
    R_inv[j][j] = 1 / R[j][j]
Q_inv = Q.transpose()

print('Q matrix:')
a = pd.DataFrame(Q)
print(a.round(2))
print('R matrix:')
b = pd.DataFrame(R)
print(b.round(2))
print('Inverse matrix:')
a = pd.DataFrame(R_inv @ Q_inv)
print(a.round(2))
# print('Origin matrix:')
# c = pd.DataFrame(Q @ R)
# print(c.round(2))

# Power Iteration
def eigenvalue(A, v):
    Av = A @ v.transpose()
    return Av.transpose() @ v

np.random.seed(0)
v = np.random.rand(n)
v /=np.linalg.norm(v)
ev = eigenvalue(data, v)

while True:
    x_new = data.dot(v)
    v_new = x_new / np.linalg.norm(x_new)
    ev_new = eigenvalue(data, v_new)
    if np.abs(ev - ev_new) < 0.0000001:
        break
    v = v_new
    ev = ev_new
print('Eigen value: {}'.format(ev_new))
print('Eigen vector: {}'.format(v_new))

# QR factorization to find eigenvalue
def QR(A):
    n, _ = A.shape
    Q = np.zeros((n,n), dtype=np.float64)
    R = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        Q[:, i] = A[:,i]
        for j in range(i):
            R[j][i] = Q[:, j].dot(A[:, i])
            Q[:, i] -= R[j][i]*Q[:, j]
        R[i][i] = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i][i]
    return Q, R

Ak = data.copy()
Q_product = np.eye(n, dtype=np.float64)
real_ev = set()
for i in range(100):
    d_old = Ak.diagonal()
    Q, R = QR(Ak)
    Q_product = Q_product @ Q
    Ak = R @ Q
    d_new = Ak.diagonal()
    for j in range(n):
        if np.abs(d_new - d_old)[j] < 0.001:
            real_ev.add(j)

real_ev = sorted(real_ev)
print('Eigen Vectors matrix:')
a = pd.DataFrame([Q_product[:, i] for i in real_ev])
print(a.T.round(2))
print('Eigen Values matrix:')
a = pd.DataFrame([Ak[i][i] for i in real_ev])
print(a.round(2))

# w, v = np.linalg.eig(data)
# print('Eigen Vectors matrix:')
# a = pd.DataFrame(v)
# print(a.round(2))
# print('Eigen Values matrix:')
# a = pd.DataFrame(w)
# print(a.round(2))
