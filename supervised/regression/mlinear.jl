using PyPlot
using Statistics
using LinearAlgebra


# Synthetic training data
N = 100

a0 = 1.29
a1 = 1.43
a2 = 1.15
a = [a0, a1, a2]

X = zeros(N,3)
for i in 1:N
    x1 = randn()
    x2 = x1*a2 + randn()/2
    X[i,:] = [1 x1 x2]
end
x1 = X[:,1]
x2 = X[:,2]

y = X*a + randn(N)/2

# Regression
lbd = 0.1     # <- Ridge
Gram = X'*X + I*lbd
MP = inv(Gram)*X'
a = MP*y
println(a)

# Quality
ry = X*a
r2 = cor(ry,y)
println("R2 = ",100*r2," %")

# Predict
for i in 1:N
    px = -2 + 4*(i-1)/99
    py = px
    X[i,:] = [1 px py]
end
pz = X*a
px = X[:,1]
py = X[:,2]

# Plot
p1 = scatter3D([],0)
fig = figure()
ax = fig.add_subplot(projection="3d")
y = [xi[1] for xi in y]
pz = [zi[1] for zi in pz]

ax.plot3D(x1,x2,y,".",color="darkgray")
ax.plot3D(px,py,pz,color="gray")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
show()
