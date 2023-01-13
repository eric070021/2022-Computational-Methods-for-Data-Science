import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

data = np.array([[87, 114, 103],
                [81,109,97],
                [73,92,88],
                [83,110,102],
                [92,109,111],
                [95,112,108],
                [83,117,106],
                [78,115,104],
                [90,101,107],
                [81,103,102],
                [87,109,103],
                [89,110,108]], dtype=np.float64)

# preprocess data
data_p = data.transpose().copy()

# step 1
for i in range(3):
    mean = np.mean(data_p[i])
    for j in range(12):
        data_p[i][j] -= mean

print('Preprocessed step 1 data:')
a = pd.DataFrame(data_p)
print(a.round(2))

# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Original data")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.scatter(data_p[0], data_p[1], data_p[2])
# plt.show()

# step 2
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

covariance = data_p @ data_p.transpose()
ev, v = QRalgo(covariance, 3)

data_p = v.transpose() @ data_p
print('Preprocessed step 2 data:')
a = pd.DataFrame(data_p)
print(a.round(2))

# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Projection onto the PCA space")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# ax.scatter(data_p[0], data_p[1], data_p[2])
# plt.show()

# step 3
data_p = np.diag(np.power(ev, -0.5)) @ data_p
print('Preprocessed step 3 data:')
a = pd.DataFrame(data_p)
print(a.round(2))

# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("After whitening")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# ax.scatter(data_p[0], data_p[1], data_p[2])
# plt.show()

# 4d
