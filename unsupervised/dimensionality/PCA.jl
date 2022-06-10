using PyPlot
using Statistics
using LinearAlgebra

# Random data
d = 3
N = 100

x = randn(N)'
y = [0.25*xi + 0.13 + 0.1*(randn()-0.5) for xi in x]
z = [0.37*xi + 0.74 + 0.1*(randn()-0.5) for xi in x]

R = vcat(x,y,z)

# Normalization
R = (R .- mean(R,dims=2))./std(R,dims=2)

# Correlation matrix
C = R*R'/(N-1)

# Eigenstates
va = eigvals(C)
ve = eigvecs(C)

# Plot data and axes
if (false)
    pygui(true)
    fig = figure()
    ax = fig.gca(projection="3d")
    ax.scatter(R[1,:],R[2,:],R[3,:])

    mx = mean(R[1,:])
    my = mean(R[2,:])
    mz = mean(R[3,:])

    ax.quiver([mx],[my],[mz],[ve[1,1]],[ve[2,1]],[ve[3,1]],color="red")
    ax.quiver([mx],[my],[mz],[ve[1,2]],[ve[2,2]],[ve[3,2]],color="red")
    ax.quiver([mx],[my],[mz],[ve[1,3]],[ve[2,3]],[ve[3,3]],color="red")
    legend(["","v_1","v_2","v_3"])
    show()
end

# Plot explained variance
if (false)
    R = rand(100,100)
    R = (R .- mean(R,dims=2))./std(R,dims=2)
    C = R*R'/99
    va = eigvals(C)
    
    va = reverse(va)
    cs = sum(va)
    PEV = 100*va./cs
    CPEV = 100*cumsum(va)./cs
    
    plot(PEV,"o-")
    xlabel("Eigenvalue number")
    ylabel("PEV [%]")
    
    pl = twinx()
    plot(CPEV,"s-",color="orange")
    ylabel("CPEV [%]")
    show()
end

# Plot projected data
if (true)
    W = hcat(ve[:,1],ve[:,2])
    R = W'*R
    scatter(R[1,:],R[2,:])
    show()
end
