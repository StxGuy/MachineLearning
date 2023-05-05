using PyPlot

# Create some random function
function random_walk()
    Y = []

    y = 0
    for i in 1:200
        y += randn()
        
        push!(Y,y)
    end
    
    return Y
end

# Simulated annealing with Metropolis-Hastings
function MH(f)
    β = 1E-3
    N = length(f)
    x = rand(1:N)
    
    for j = 1:1000
        for i = 1:(10*N)
            y = rand(max(1,x-3):min(x+3,N))
            A = min(1,exp(β*(f[y]-f[x])))
            
            if (rand() ≤ A)
                x = y
            end
        end
        
        β += 1E-3
    end
    
    return x
end

# Main function
f = random_walk()
println("Found: ", MH(f),", Should be: ",argmax(f))
plot(f)
show()
