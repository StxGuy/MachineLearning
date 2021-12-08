import numpy as np
import matplotlib.pyplot as pl

NSamples = 100

# Generate synthetic data
x = [np.random.randn() for i in range(NSamples)]
y = [np.random.randn() for i in range(NSamples)]

x = np.append(x,[5 + np.random.randn() for i in range(NSamples)])
y = np.append(y,[5 + np.random.randn() for i in range(NSamples)])

x = np.append(x,[-1 + np.random.randn() for i in range(NSamples)])
y = np.append(y,[6 + np.random.randn() for i in range(NSamples)])

R = np.c_[x,y]

# k-means
C = [[0,0],[3,3],[0,4]]             # Centroids
z = [0 for i in range(3*NSamples)]  # Labels

for loop in range(4):
    # Clusterize points
    for j in range(3*NSamples):
        d_min = 10
        for i in range(3):
            d = np.linalg.norm(R[j]-C[i])
            if (d < d_min):
                d_min = d
                i_min = i
        z[j] = i_min

    # Recalculate centroids
    C = [[0,0],[0,0],[0,0]]
    s = [0,0,0]
    for j in range(3*NSamples):
        C[z[j]] = C[z[j]] + R[j]
        s[z[j]] = s[z[j]] + 1
        
    for i in range(3):
        C[i] = C[i]/s[i]

# Print results
for c in C:
    print(c)

pl.subplot(121)        
pl.scatter(x,y)
pl.subplot(122)
pl.scatter(x,y,c=z)

# Silhouette


pl.show()
