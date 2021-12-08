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
lst = []

r2 = 0
r1a = []
r2a = []
for it in range(10000):
    rho = [np.dot(e,r), np.dot(x1,r), np.dot(x2,r)]
    i = np.argmax(np.absolute(rho))
    
    if i not in lst:    
        lst.append(i)
    
    # Find equiangular direction
    d = 0
    for j in lst:
        d = d + eta*np.sign(rho[j])
    d = d / len(lst)
    
    # Update all coefficients in the list and residuals
    for j in lst:
        a[j] = a[j] + d
        r = r - d*X[:,j]
          
    x.append(np.linalg.norm(a))
    y1.append(a[0])
    y2.append(a[1])
    y3.append(a[2])
    
    yh = np.matmul(X,a)
    r2 = np.corrcoef(yh,np.transpose(r))[0][1]
    r2a.append(r2)
    r1a.append(1-np.dot(r,r)/np.matmul(np.transpose(y),y)[0][0])
    
print(a)
print(r2)

pl.subplot(121)
pl.plot(x,y1,color='lightgray')
pl.plot(x,y2,color='gray')
pl.plot(x,y3,color='darkgray')
pl.axis([0,2.5,0,1.6])
pl.xlabel('|a|')
pl.ylabel('a')

pl.subplot(122)
pl.plot(x,r1a,color='gray')
pl.xlabel('Step')
pl.ylabel('R$^2$')
pl.axis([0,2.5,0,1])
#pl.savefig('../figs/src/LARS.svg')

pl.show()
        
        
