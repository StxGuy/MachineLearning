using LinearAlgebra

function Θ(x)
    if (x >= 0)
        return 1
    else
        return 0
    end
end


function train(w,b)
    example = [ 0 0 0;
                0 1 1;
                1 0 1;
                1 1 1;]
    η = 0.001    
    ε = 1
    while(ε > 0)
        ε = 0
        for epoch in 1:64
            r = rand(1:4)
            x = example[r,1:2] + rand(2)/150
            f = example[r,3]
            f_hat = Θ(w⋅x+b)
            
            # Update
            Δf = f_hat-f
            w = w - η*Δf*x
            b = b - η*Δf
            ε += abs(Δf)
        end
        ε /= 64
            
        println(ε)
    end
    
    return w,b
end

w,b = train(rand(2).-0.5,rand()-0.5)

println("0 0 -> ", Θ(w⋅[0 0] + b))
println("0 1 -> ", Θ(w⋅[0 1] + b))
println("1 0 -> ", Θ(w⋅[1 0] + b))
println("1 1 -> ", Θ(w⋅[1 1] + b))

