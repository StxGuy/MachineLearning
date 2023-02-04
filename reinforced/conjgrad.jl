using LinearAlgebra

#f(x,y) = 100*(y-x^2)^2+(1-x)^2

#x = 0
#y = 0

function c∇(A,b,x)
    r = b - A*x
    p = r
    ξ = r⋅r
    
    for i in 1:5
        α = ξ/(p⋅(A*p))
        x = x + α*p
        r = r - α*A*p
        new_ξ = r⋅r
                
        if (new_ξ < 1E-10)
            break
        else
            p = r + (new_ξ/ξ)*p
            ξ = new_ξ
        end
    end
    
    
    return x
end


A = [4 1;1 3]
b = [1; 2]

x = [2; 1]

y = c∇(A,b,x)

println(y)
        
        
