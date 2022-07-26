using PyPlot
using LinearAlgebra

# Synthetic training data
function createData(N)
    e1 = randn(N)
    e2 = randn(N).+3
    f1 = 2*e1.+1 + randn(N)
    f2 = 3*e2 + randn(N)

    y = vcat(zeros(N),ones(N))
    x1 = vcat(e1,e2)
    x2 = vcat(f1,f2)
    ve = ones(2*N)
    X = hcat(ve,x1,x2)

    return X,x1,x2,y
end

# Regression
function regression(X,y)
    N = length(y)
    w = [rand(),rand(),rand()]
    p = zeros(N)

    for it in 1:N
        p = [1.0/(1.0 + exp(-w⋅r)) for r in eachrow(X)]
        S = diagm(p)

        w = w + inv(X'*S*X)*X'*(y-p)
    end

    # Quality
    L = 1
    for (qi,yi) in zip(p,y)
        L = L*(qi^yi)*((1-qi)^(1-yi))
    end

    println(-2*log(L))

    return w
end

# Make a prediction
function predict(w)
    # Predict & Plot
    p = LinRange(-2,7,100)
    q = LinRange(-2,16,100)
    Z = zeros((100,100))

    for i in 1:100
        for j in 1:100
            r = [1,p[i],q[j]]
            Z[i,j] = 1.0/(1.0 + exp(-w⋅r))
        end
    end

    return p,q,Z
end

# Numpy-like meshgrid function
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

#========================================#
#                 MAIN                   #
#========================================#
X,x1,x2,y = createData(100)
w = regression(X,y)
p,q,Z = predict(w)
X,Y = meshgrid(p,q)

# Plot results
#p1 = scatter3D([],0)
fig = figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X,Y,Z)

ax.plot3D(x1,x2,y,".",color="gray")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("p")
ax.view_init(21,-61)
show()


