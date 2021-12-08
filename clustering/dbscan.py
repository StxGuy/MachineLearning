import matplotlib.pyplot as pl
import numpy as np

class dbscan:
    def __init__(self,n):
        self.NPoints = n
        self.xv = []
        self.yv = []
        self.zv = []
        
        for i in range(n):
            ro = 1
            if np.random.rand() > 0.5:
                ro = ro + 1
            r = ro + 2*(np.random.rand()-0.5)/10

            t = (np.random.rand()-0.5)*2*np.pi
            x = r*np.cos(t)
            y = r*np.sin(t)
            
            self.xv.append(x)
            self.yv.append(y)
            self.zv.append(0)
            
    def neighborhood(self,j):
        N = []
        for i in range(self.NPoints):
            d = np.sqrt((self.xv[j]-self.xv[i])**2 + (self.yv[j]-self.yv[i])**2)
            if (d < 0.5):
                N.append(i)
        
        return N
    
    # Private
    def singleshot(self,i,C):
        visit = []
        visit.append(i)
        while(len(visit) > 0):
            el = visit.pop()
            
            v = self.neighborhood(el)
            if (len(v) > 3 and self.zv[el] == 0):
                self.zv[el] = C
                
                for k in v:
                    visit.append(k)

    def clusterize(self):
        C = 0
        for i in range(self.NPoints):
            if (self.zv[i] == 0):
                v = self.neighborhood(i)
                if (len(v) > 3):
                    C = C + 1
                    self.singleshot(i,C)
                
    def plot(self):
        pl.scatter(self.xv,self.yv,c=self.zv)
                

d = dbscan(300)
d.clusterize()
d.plot()
pl.show()

#def neighborhood(x,y):
    

#NClusters = 0

#for i in range(NPoints):
    #if (z[i] == 0):
        #lista.append((xv[i],yv[i]))
    
    #x,y = lista.pop()
    #V = neighborhood(x,y)
