import numpy as np
import pandas as pd
from math import sqrt

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

# d
def standardize(data, n):
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

def QRalgo(data, n):
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

    Ak = data
    Q_product = np.eye(n, dtype=np.float64)
    for i in range(100):
        Q, R = QR(Ak)
        Q_product = Q_product @ Q
        Ak = R @ Q
    return Ak.diagonal(), Q_product
data_n = data.copy()
data_n = standardize(data_n, n)
# do SVD on data/sqrt(n-1)
ATA = (data_n.transpose() @ data_n) / (n - 1)

s_square, v = QRalgo(ATA, n)
s = s_square.copy()
for i in range(n):
    if s_square[i] > 0:
        s[i] = sqrt(s_square[i])
u = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    if s[i] > 0:
        u[:, i] = ((data_n/sqrt(n-1)) @ v[:, i]) / s[i]

print('Principle component matrix:')
a = pd.DataFrame(v)
print(a.round(2))
print('Eigen Values matrix:')
a = pd.DataFrame(s_square)
print(a.round(2))

# rank 3 approximate of data
data_rank3 = data.copy()
ATA = data_rank3.transpose() @ data_rank3
s_square, v = QRalgo(ATA, n)
s = s_square.copy()
for i in range(n):
    if s_square[i] > 0:
        s[i] = sqrt(s_square[i])
u = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    if s[i] > 0:
        u[:, i] = (data_rank3 @ v[:, i]) / s[i]

rank3_s = np.zeros((n, m), dtype=np.float64)
for i in range(3):
    rank3_s[i][i] = s[i]
rank3_data = u @ rank3_s @ v.transpose()
print('Rank 3 approximation of data:')
a = pd.DataFrame(rank3_data)
print(a.round(2))
