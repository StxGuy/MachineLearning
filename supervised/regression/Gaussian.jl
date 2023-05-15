using LinearAlgebra
using Distributions
using PyPlot


function kernel(x,y)
    σₒ = 1.0    # signal variance
    λ = 0.6   # length scale
    
    L2(x) = sum(x.^2)
    
    Nx = length(x)
    Ny = length(y)
    
    K = Array{Float64}(undef,Nx,Ny)
    
    for i in 1:Nx
        for j in 1:Ny
            K[i,j] = σₒ*exp(-L2(x[i]-y[j])/2λ^2)
        end
    end
    
    return K
end

N = 10
M = 100

x = 2π*rand(N)
xₒ = LinRange(0,2π,M)
f = sin.(x)

μ = kernel(xₒ,x)*inv(kernel(x,x))*f
Σ = kernel(xₒ,xₒ)-kernel(xₒ,x)*inv(kernel(x,x)+1E-6*I)*kernel(x,xₒ)

L = cholesky(Symmetric(Σ),check=false).L

for i in 1:15
    g = μ + L*randn(M)
    plot(xₒ,g,alpha=0.3)
end

plot(xₒ,μ)
plot(x,f,"*",color="black")
show()


