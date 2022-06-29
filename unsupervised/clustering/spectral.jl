using PyPlot
using LinearAlgebra

function createdata(N)
    R = []
    for i in 1:N
        r = 5+rand()*2
        θ = 2*(rand()-0.5)*π
        x = r*cos(θ)
        y = r*sin(θ)
    
        if (i == 1)
            R = [x y 0]
        else
            R = vcat(R,[x y 0])
        end
        
        x = randn()
        y = randn()
        R = vcat(R,[x y 0])        
    end
    
    return R
end

function Euclidean(x,y)
    return sum((x-y).^2)
end

function spec(R)
    N,M = size(R)
    
    A = zeros(N,N)
    D = zeros(N,N)
    
    σ² = 1
    
    # Create adjacency and degree matrices
    for i in 1:(N-1)
        for j in (i+1):N
            d = Euclidean(R[i,1:2],R[j,1:2])
            A[i,j] = exp(-d/(2σ²))
            A[j,i] = A[i,j]
        end
    end
    for i in 1:N
        D[i,i] = sum(A[i,:])
    end
        
    # Find second eigenvector
    K = I-inv(D)*A
    Λ = eigvals(K)
    z = eigvecs(K)[:,2]
    
    # Cluster
    return sign.(z)
end               

R = createdata(200)
z = spec(R)
scatter(R[:,1],R[:,2],10,z)

show()
