import numpy as np
import matplotlib.pyplot as pl

NSamples = 100

if (False):
    # Generate synthetic data
    x = [np.random.randn() for i in range(NSamples)]
    y = [np.random.randn() for i in range(NSamples)]

    x = np.append(x,[5 + np.random.randn() for i in range(NSamples)])
    y = np.append(y,[5 + np.random.randn() for i in range(NSamples)])

    x = np.append(x,[-1 + np.random.randn() for i in range(NSamples)])
    y = np.append(y,[6 + np.random.randn() for i in range(NSamples)])

    R = np.c_[x,y]

    f = open("data.dat","w")

    for ex,ey in zip(x,y):
        f.write(str(ex)+"\t"+str(ey))
        f.write("\n")
        
    f.close()
else:
    f = open("data.dat","r")
    g = open("labels.dat","r")
    h = open("silhouette.dat","r")
    
    x = []
    y = []
    z = []
    t = []
    for i in range(300):
        line = f.readline()
        row = line.split()
        
        x.append(float(row[0]))
        y.append(float(row[1]))
        
        line = g.readline()
        row = line.split()
        
        z.append(int(row[0]))
        
        line = h.readline()
        row = line.split()
        t.append(float(row[0]))
    f.close()
    g.close()
    
    pl.subplot(221)
    pl.scatter(x,y,color='gray')
    pl.subplot(222)
    pl.scatter(x,y,c=z)
    pl.subplot(223)
    #pl.bar(np.arange(len(t)),t,width=1.0,color='gray')
    pl.plot(t,color='gray')
    pl.axis([0,300,0,1.25])
    #pl.savefig("clusters.svg")
    pl.show()
