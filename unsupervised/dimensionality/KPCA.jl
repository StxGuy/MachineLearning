using PyPlot
using LinearAlgebra
using Statistics

function nonlineardata(N)
    R = []
    c = []
    for θ in LinRange(0,π,N÷2)
        x = cos(θ)
        y = sin(θ)
        if (length(R) == 0)
            R = [x;y]
        else
            R = hcat(R,[x;y])
        end
        if (length(c) == 0)
            c = [0.5 0.5 0.5]
        else
            c = vcat(c,[0.5 0.5 0.5])
        end            
        
        x = 1 + cos(θ+π)
        y = 0.5 + sin(θ+π)
        R = hcat(R,[x;y])
        c = vcat(c,[0.25 0.25 0.25])
    end
    
    return R,c
end


R,c = nonlineardata(100)
scatter(R[1,:],R[2,:],10,c)
xlabel("x")
ylabel("y")
axis("equal")
show()

const γ = 15
kernel(x,y) = exp(-γ*sum((x - y).^2))

function kernelPCA(X,kernel)
    N = size(X,2)
    
    K = [kernel(X[:,i],X[:,j]) for i in 1:N,j in 1:N]

    # recenter K
    J = ones(N,N)
    Kp = (I-J/N)*K*(I-J/N)

    # eigendecomposition 
    v = eigvals(Kp)
    A = eigvecs(Kp)

    A = reverse(A,dims=2)
    v = reverse(v)
    v[v .< 0] .= 0
    Σ = Diagonal(sqrt.(v))

    # Projection
    nΦ = Σ*A'

    return nΦ
end

# Main and plotting
Φ = kernelPCA(R,kernel)
scatter(Φ[1,:],Φ[2,:],10,c)
xlabel("Phi_x")
ylabel("Phi_y")
axis("equal")
show()

