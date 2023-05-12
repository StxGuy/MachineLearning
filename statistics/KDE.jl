using Statistics
using PyPlot

function kernel(u,k)
    if (abs(u) < 1)
        if (k == "Uni")
            return 0.5
        end
        if (k == "Epa")
            return 0.75*(1-u^2)
        end
        if (k == "Biw")
            return (15.0/16)*(1-u^2)^2
        end
        if (k == "Tri")
            return (35.0/32)*(1-u^2)^3
        end
    else
        return 0
    end
end

function KDE(x,Nw,k)
    mi = minimum(x)
    ma = maximum(x)
    N = length(x)
    
    h = 1.06*std(x)*N^(-0.2)
    
    x′ = LinRange(mi,ma,Nw)
    
    z = []
    for xo in x′
        s = 0
        for xi in x
            u = (xi-xo)/h
            s += kernel(u,k)
        end
        push!(z,s/(N*h))
    end
    
    return x′,z#/sum(z)
end

function normal(x,μ,σ²)
    return exp(-(x-μ)^2/(2σ²))/sqrt(2π*σ²)
end    

μ = 3.0
σ² = 1.7

f = 3.0 .+ sqrt(σ²)*randn(2000)
x,y = KDE(f,50,"Biw")
z = normal.(x,μ,σ²)

plot(x,y,"s")
plot(x,z)
show()
    
