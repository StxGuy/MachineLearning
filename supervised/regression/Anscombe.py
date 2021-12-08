import matplotlib.pyplot as pl
import numpy as np

X = [   [10,8,13,9,11,14,6, 4,12,7,5],
        [10,8,13,9,11,14,6, 4,12,7,5],
        [10,8,13,9,11,14,6, 4,12,7,5],
        [8, 8, 8,8, 8, 8,8,19, 8,8,8]]

Y = [   [8.04,6.95, 7.58,8.81,8.33,9.96,7.24, 4.26,10.84,4.82,5.68],
        [9.14,8.14, 8.74,8.77,9.26,8.10,6.13, 3.10, 9.13,7.26,4.74],
        [7.46,6.77,12.74,7.11,7.81,8.84,6.08, 5.39, 8.15,6.42,5.73],
        [6.58,5.76, 7.71,8.84,8.47,7.04,5.25,12.50, 5.56,7.91,6.89]]


c = ['.','s','*','^']

for n in range(4):
    a = np.cov(X[n],Y[n])[0][1]/np.var(X[n])
    b = np.mean(Y[n])-a*np.mean(X[n])
    x = np.linspace(3,20,100)
    y = [a*xi + b for xi in x]
    
    print('Sample #'+str(n))
    print('Mean   x: '+str(np.mean(X[n])))
    print('Mean   y: '+str(np.mean(Y[n])))
    print('Var    x: '+str(np.var(X[n])))
    print('Var    y: '+str(np.var(Y[n])))
    print('cor(x,y): '+str(np.corrcoef(X[n],Y[n])[0][1]))
    print('Line    : '+str(a)+'x + '+str(b))
    print('--------------------------------')

    pl.subplot(221 + n)    
    pl.plot(x,y,color='lightgray')
    pl.plot(X[n],Y[n],c[n],color='gray')
    pl.axis([3,20,2,14])
    pl.xlabel('x1')
    pl.ylabel('y1')
    
pl.show()    



