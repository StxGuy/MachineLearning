import numpy as np

Data = np.array([[5.9E7,   400.0, 3.0, 118.4, 0],
                 [6.2E7,   430.0, 2.5, 125.6, 0],
                 [1.7E7,   120.0, 2.5,   0.0, 0],
                 [7.1E6,    29.0, 1.5,  28.9, 0],
                 [3.3E6,    50.0, 3.5,  78.0, 0],
                 [1.0E3,   150.0, 6.5, 133.6, 1],
                 [1.0E-15, 0.205, 2.0, 200.0, 0],
                 [2.0E3,    60.0, 6.0, 119.0, 1]])

Data = np.array([[10,  10,0],
                 [10, 400,1],
                 [1E7, 10,0],
                 [1E7,400,0],
                 [1E3, 60,1],
                 [1E3, 60,1]])

nr,nc = np.shape(Data)
p0 = np.sum(Data[:,nc-1])/nr
p1 = (nr-np.sum(Data[:,nc-1]))/nr
Ho = -p0*np.log2(p0)-p1*np.log2(p1)

print(Ho)

def condS(Data,col,X):
    X = np.array(X)
    N = np.size(X,0)
    M = np.zeros((N+1,5))
    nr,nc = np.shape(Data)
    
    print(X[1,0])
        
    for a,b in zip(Data[:,col],Data[:,nc-1]):
        for i in range(N):
            if (a >= X[i,0] and a <= X[i,1]):
                if (b == 1):
                    M[i,0] = M[i,0] + 1
                else:
                    M[i,1] = M[i,1] + 1

    for i in range(N):
        M[i,2] = np.sum(M[i,:])

    for i in range(3):
        M[N,i] = np.sum(M[:,i])

X = condS(Data,0,[[0,100],[200,1E5],[1E6,1E8]])        
print(X)      
