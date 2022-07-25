using PyPlot

# Synthetic training data
N = 100
a = 1.5
b = 2.0
x = randn(N)
y = a*sin.(x*b) + randn(N)/8

# Levenberg-Marquardt
w = [0.7,1.1]
for it in 1:10
    J1 = sin.(w[2]*x)
    J2 = w[1]*x.*cos.(w[2]*x)
    J = hcat(J1,J2)

    f = w[1]*sin.(x*w[2])

    global w = w + (inv(J'*J)*J')*(y-f)
end

# Predict
px = LinRange(-3,3,100)
py = w[1]*sin.(px*w[2])

println(w)

plot(px,py,color="lightgray")
plot(x,y,".",color="gray")
show()
