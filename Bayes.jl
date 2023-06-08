using PyPlot

function muvar(M)
    N, = size(M)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    
    # First pass: mean
    μₚʸ = 0
    μₚⁿ = 0
    μᵥʸ = 0
    μᵥⁿ = 0
    n = 0
    
    for i in 1:N
        if (M[i,3] == 1)
            push!(x1,M[i,1])
            push!(x2,M[i,2])
            
            μₚʸ += M[i,1]
            μᵥʸ += M[i,2]
            n += 1
        else
            push!(y1,M[i,1])
            push!(y2,M[i,2])
            
            μₚⁿ += M[i,1]
            μᵥⁿ += M[i,2]
        end
    end
    
    μₚʸ /= n
    μₚⁿ /= (N-n)
    μᵥʸ /= n
    μᵥⁿ /= (N-n)
    
    # Second pass: variance
    σ²ₚʸ = 0
    σ²ₚⁿ = 0
    σ²ᵥʸ = 0
    σ²ᵥⁿ = 0
    
    for i in 1:N
        if (M[i,3] == 1)
            σ²ₚʸ += (M[i,1] - μₚʸ)^2
            σ²ᵥʸ += (M[i,2] - μᵥʸ)^2
        else
            σ²ₚⁿ += (M[i,1] - μₚⁿ)^2
            σ²ᵥⁿ += (M[i,2] - μᵥⁿ)^2
        end
    end
    
    σ²ₚʸ /= (n-1)
    σ²ₚⁿ /= (N-n-1)
    σ²ᵥʸ /= (n-1)
    σ²ᵥⁿ /= (N-n-1)
    
    return [μₚʸ,μₚⁿ,μᵥʸ,μᵥⁿ,σ²ₚʸ,σ²ₚⁿ,σ²ᵥʸ,σ²ᵥⁿ]
end

function Gaussian(x,μ,σ²)
    return exp(-(x-μ)^2/(2σ²))/sqrt(2π*σ²)
end

function Classifier(price,volume,p)
    μₚʸ,μₚⁿ,μᵥʸ,μᵥⁿ,σ²ₚʸ,σ²ₚⁿ,σ²ᵥʸ,σ²ᵥⁿ = p
    
    P_buy = Gaussian(price,μₚʸ,σ²ₚʸ)*Gaussian(volume,μᵥʸ,σ²ᵥʸ)
    P_not = Gaussian(price,μₚⁿ,σ²ₚⁿ)*Gaussian(volume,μᵥⁿ,σ²ᵥⁿ)
    
    if P_buy > P_not
        return 1
    else
        return 0
    end
end

M = [10.3 100 0;
     9.5 50 1;
     11.5 90 0;
     5.5 30 1;
     7.5 120 1;
     12.2 40 0;
     7.1 80 1;
     10.5 65 0]

N, = size(M)
x1 = []
x2 = []
y1 = []
y2 = []

for i in 1:N
    if (M[i,3] == 1)
        push!(x1,M[i,1])
        push!(x2,M[i,2])
    else
        push!(y1,M[i,1])
        push!(y2,M[i,2])
    end
end

par = muvar(M)

p_sp = LinRange(5.2,12.5,100)
v_sp = LinRange(25.6,124.4,100)
K = zeros(100,100)

for i in 1:100
    p = p_sp[i]    
    for j in 1:100
        v = v_sp[j]
        K[j,i] = Classifier(p,v,par)
    end
end
        
contourf(p_sp,v_sp,K,alpha=0.5)

# plotting
#println("Buy")
#println(" mu_p: ",μₚʸ,", var_p: ",σ²ₚʸ,", mu_v: ",μᵥʸ,", var_v: ",σ²ᵥʸ)
#println("Don't")
#println(" mu_p: ",μₚⁿ,", var_p: ",σ²ₚⁿ,", mu_v: ",μᵥⁿ,", var_v: ",σ²ᵥⁿ)

plot(x1,x2,"s",color="blue")
plot(y1,y2,"o",color="orange")
xlabel("Price")
ylabel("Volume")
show()
