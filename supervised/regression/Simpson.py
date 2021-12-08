import matplotlib.pyplot as pl
import numpy as np

xo = [0,1,2,3]
yo = [30,20,10,0]

xl = []
yl = []

for n in range(4):
    x = np.random.randn(100)
    y = np.array([2*xi + np.random.randn() for xi in x])

    xd = x + xo[n]
    yd = y + yo[n]
    
    xl = np.append(xl,xd)
    yl = np.append(yl,yd)

    xf = np.linspace(-4,4,100)
    a = np.cov(x,y)[1][0]/np.var(x)
    b = np.mean(y)-a*np.mean(x)
    yf = np.array([a*xi + b for xi in xf])

    xf = xf + xo[n]
    yf = yf + yo[n]

    pl.plot(xf,yf,color='lightgray')
    pl.plot(xd,yd,'.',color='gray')

x = np.linspace(-4,7,100)
a = np.cov(xl,yl)[0][1]/np.var(xl)
b = np.mean(yl) - a*np.mean(xl)
y = np.array([a*xi + b for xi in x])
pl.plot(x,y,':',color='darkgray')
pl.axis([-6,8,-10,40])
pl.xlabel('x')
pl.ylabel('y')
    
pl.show()
