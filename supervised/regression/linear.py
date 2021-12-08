import matplotlib.pyplot as pl
import numpy as np


# Synthetic training data
N = 100
a = 1.5
b = 1.0
x = [np.random.randn() for i in range(N)]
y = [a*xi + b + np.random.randn()/2 for xi in x]

# Linear Regression
a = np.cov(x,y)[0][1]
b = np.mean(y) - a*np.mean(x)
nx = np.linspace(-2,2,100)
ny = [a*xi + b for xi in nx]

# Quality of regression
ry = [a*xi + b for xi in x]
r2 = np.corrcoef(y,ry)[0][1]**2
print("R2 = "+str(100*r2)+" %")

# Plot
pl.plot(x,y,'.',color='gray')
pl.plot(nx,ny,color='lightgray')
pl.axis([-3,3,-3,4])
pl.xlabel('x')
pl.ylabel('y')
#pl.savefig('../figs/src/linear.svg')
pl.show()

