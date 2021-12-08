import matplotlib.pyplot as pl
import numpy as np


# Synthetic training data
N = 100

a0 = 1.29
a1 = 1.43
a2 = 1.15
a = [[a0],[a1],[a2]]

x1 = np.array([np.random.randn() for i in range(N)])
x2 = np.array([np.random.randn() for i in range(N)])
e = [1 for i in range(N)]
X = np.c_[e,x1,x2]

y = np.dot(X,a) + np.random.randn(N,1)


# Regression
eta = 0.05
a = [0,0,0]
r = [yi[0] for yi in y]

y1 = []
y2 = []
y3 = []
x = []

r2 = 0
for it in range(15000):
    rho = [np.dot(e,r), np.dot(x1,r), np.dot(x2,r)]
    i = np.argmax(np.absolute(rho))
    
    d = eta*np.sign(rho[i])
    a[i] = a[i] + d
    r = r - d*X[:,i]
          
    x.append(np.linalg.norm(a))
    y1.append(a[0])
    y2.append(a[1])
    y3.append(a[2])
    
    yh = np.matmul(X,a)
    r2 = np.corrcoef(yh,np.transpose(y))[0][1]
    
print(a)
print(r2)

pl.plot(x,y1)
pl.plot(x,y2)
pl.plot(x,y3)

pl.show()
        
        
