import matplotlib.pyplot as pl
import numpy as np


# Synthetic training data
N = 100

a0 = 1.29
a1 = 1.43
a2 = 1.15
a = [[a0],[a1],[a2]]

x1 = [np.random.randn() for i in range(N)]
x2 = [ex*a2 + np.random.randn()/2 for ex in x1]
e = [1 for i in range(N)]
X = np.c_[e,x1,x2]

y = np.dot(X,a) + np.random.randn(N,1)/2

# Regression
lbd = 0.1     # <- Ridge
Gram = np.dot(np.transpose(X),X) + np.eye(3)*lbd
MP = np.dot(np.linalg.inv(Gram),np.transpose(X))
a = np.dot(MP,y)
print(a)

# Quality
ry = np.dot(X,a)
r2 = np.corrcoef(np.transpose(ry),np.transpose(y))[0][1]
print("R2 = "+str(100*r2)+" %")

# Predict
px = np.linspace(-2,2,100)
py = np.linspace(-2,2,100)
X = np.c_[e,px,py]
pz = np.dot(X,a)

# Plot
y = [xi[0] for xi in y]
pz = [zi[0] for zi in pz]

fig = pl.figure()
ax = pl.axes(projection='3d')
ax.plot3D(x1,x2,y,'.',color='darkgray')
ax.plot3D(px,py,pz,color='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.axes.set_xlim3d(left=-4,right=4)
ax.axes.set_ylim3d(bottom=-4,top=4)
ax.axes.set_zlim3d(bottom=-6,top=8)
#pl.savefig('../figs/src/mlinear.svg')
pl.show()
