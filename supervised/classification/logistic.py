import matplotlib.pyplot as pl
import numpy as np

# Synthetic training data
N = 100

e1 = np.random.randn(N)
f1 = [2*xi+1 + np.random.randn() for xi in e1]
e2 = np.random.randn(N)+3
f2 = [3*xi+0.5 + np.random.randn() for xi in e2]
y = np.append(np.zeros(N),np.ones(N))

x1 = np.append(e1,e2)
x2 = np.append(f1,f2)
e = [1 for i in range(2*N)]

X = np.c_[e,x1,x2]

# Regression
w = [np.random.rand(),np.random.rand(),np.random.rand()]

for it in range(200):
    p = [1.0/(1.0 + np.exp(-np.dot(w,r))) for r in X]
    S = np.diag(p)

    G = np.matmul(np.matmul(np.transpose(X),S),X)
    D = np.matmul(np.linalg.inv(G),np.transpose(X))
    w = w + np.matmul(D,y-p)
    
# Quality
L = 1
for pi,yi in zip(p,y):
    L = L*(pi**yi)*((1-pi)**(1-yi))
    
print(-2*np.log(L))
    
# Predict & Plot
p = np.linspace(-2,7,100)
q = np.linspace(-2,16,100)
X,Y = np.meshgrid(p,q)
Z = np.zeros((100,100))

for j in range(100):
    for i in range(100):
        r = [1,p[i],q[j]]
        Z[i][j] = 1.0/(1.0 + np.exp(-np.dot(w,r)))

fig = pl.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,cmap="Greys",alpha=0.7,shade=True)

ax.plot3D(x1,x2,y,'.',color='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('p')
ax.view_init(21,-61)
#pl.savefig('../figs/src/logistic.svg')
pl.show()

