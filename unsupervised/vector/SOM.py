import numpy as np
import matplotlib.pyplot as pl

grid = np.random.rand(5,5,2)
x = np.zeros(2)
tau_s = 15
tau_e = 15
sigma2 = 20
eta = 0.1
t = 0

for it in range(10000):
    # Sample
    x[0] = 2*(np.random.rand()-0.5)
    x[1] = np.sqrt(1-x[0]*x[0])
    if (np.random.rand() > 0.5):
        x[1] = -x[1]

    # Find BMU
    d_min = 1E5
    for i in range(5):
        for j in range(5):
            d = np.linalg.norm(x - grid[i,j,:])
            if d < d_min:
                d_min = d
                iw = i
                jw = j

    # Adjust weights
    for i in range(5):
        for j in range(5):
            nei = np.exp(-((i-iw)**2+(j-jw)**2)/(2*sigma2))
            grid[i,j,:] = grid[i,j,:] + eta*nei*(x - grid[i,j,:])
            
    # Evolution
    t = t + 0.00001
    sigma2 = sigma2*np.exp(-t/tau_s)
    eta = eta*np.exp(-t/tau_e)
    
for i in range(5):
    for j in range(5):
        pl.plot(grid[i,j,0],grid[i,j,1],'s')

pl.show()        
