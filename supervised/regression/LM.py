import matplotlib.pyplot as pl
import numpy as np

# Synthetic training data
N = 100
a = 1.5
b = 2.0
x = [np.random.randn() for i in range(N)]
y = np.array([[a*np.sin(xi*b)+np.random.randn()/8] for xi in x])


# Levenberg-Marquardt

w = np.array([[0.7],[1.1]])
for it in range(10):    
    J = []
    f = []
    for xi in x:
        J.append([np.sin(w[1][0]*xi),w[0][0]*xi*np.cos(w[1][0]*xi)])
        f.append([w[0][0]*np.sin(xi*w[1][0])])
    gram = np.dot(np.transpose(J),J)
    MP = np.matmul(np.linalg.inv(gram),np.transpose(J))
    w = w + np.matmul(MP,y-f)

# Predict
px = np.linspace(-3,3,100)
py = [w[0][0]*np.sin(xi*w[1][0]) for xi in px]

print(w)

pl.plot(px,py,color="lightgray")
pl.plot(x,y,'.',color="gray")
#pl.savefig("../figs/src/LM.svg")
pl.show()
