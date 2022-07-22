using PyPlot
using Statistics

# Synthetic training data
N = 100
a = 1.5
b = 1.0
x = [randn() for i in 1:N]
y = [a*xi + b + randn()/2 for xi in x]

# Linear Regression
a = cov(x,y)
b = mean(y) - a*mean(x)
nx = LinRange(-2,2,100)
ny = [a*xi + b for xi in nx]

# Quality of regression
ry = [a*xi + b for xi in x]
r2 = cor(y,ry)^2
println("R2 = ",100*r2," %")

# Plot
plot(x,y,".",color="gray")
plot(nx,ny,color="lightgray")
axis([-3,3,-3,4])
xlabel("x")
ylabel("y")
#pl.savefig('../figs/src/linear.svg')
show()
