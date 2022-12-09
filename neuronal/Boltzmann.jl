using LinearAlgebra
using PyPlot

p0 = [1 1 1;
      1 0 1;
      1 0 1;
      1 0 1;
      1 1 1]
p0 = reshape(p0,15,1)
      
p1 = [0 0 1;
      0 1 1;
      0 0 1;
      0 0 1;
      0 0 1]
p1 = reshape(p1,15,1)      
      
p2 = [1 1 1;
      0 0 1;
      0 1 0;
      1 0 0;
      1 1 1]
p2 = reshape(p2,15,1)
      
p3 = [1 1 1;
      0 0 1;
      0 1 1;
      0 0 1;
      1 1 1]
p3 = reshape(p3,15,1)
      
p4 = [1 0 1;
      1 0 1;
      1 1 1;
      0 0 1;
      0 0 1]
p4 = reshape(p4,15,1)
      
p5 = [1 1 1;
      1 0 0;
      1 1 1;
      0 0 1;
      1 1 1]
p5 = reshape(p5,15,1)
      
p6 = [1 0 0;
      1 0 0;
      1 1 1;
      1 0 1;
      1 1 1]
p6 = reshape(p6,15,1)
      
p7 = [1 1 1;
      0 0 1;
      0 1 0;
      0 1 0;
      0 1 0]
p7 = reshape(p7,15,1)
      
p8 = [1 1 1;
      1 0 1;
      0 1 0;
      1 0 1;
      1 1 1]
p8 = reshape(p8,15,1)
      
p9 = [1 1 1;
      1 0 1;
      1 1 1;
      0 0 1
      0 0 1]      
p9 = reshape(p9,15,1)

training_set = [p0 p1 p2 p3 p4 p5 p6 p7 p8 p9]


function σ(x)
   return 1/(1+exp(-x))
end

# Training
function train(b,c,W,tset)
    η = 1E-3
    N = size(W,1)
    h = zeros(N,1)
    x = zeros(N,1)
    
    ε = 0
    for s in eachcol(tset)
        # Set hidden layer
        for i in 1:N
            p = σ(c[i] + W[i,:]·s)
            if (rand() < p)
                h[i] = 1
            else
                h[i] = 0
            end
        end
        
        # Reconstruct visible layer
        for i in 1:N
            p = σ(b[i] + h·W[:,i])
            if (rand() < p)
                x[i] = 1
            else
                x[i] = 0
            end
        end
        
        # Estimate gradients
        y0 = σ.(c + W*s)
        y1 = σ.(c + W*x)
        
        ∇b = s - x
        ∇c = y0 - y1
        ∇W = s*transpose(y0) - x*transpose(y1)
        
        # Improve parameters
        b += η*∇b
        c += η*∇c
        W += η*transpose(∇W)
        
        # Calculate error
        ε += -b·s - c·h - h·(W*s)
    end
    
    ε /= size(tset,2)
        
    return b,c,W,ε
end

N = 15
c = zeros(N,1)
b = zeros(N,1)
W = randn(N,N)

for it in 1:100000
    global b,c,W,ε = train(b,c,W,training_set)
    println(ε)
end

s = [1 1 0;
     0 0 1;
     0 1 0;
     1 1 0;
     0 1 1]

s = reshape(s,15,1)     
x = reshape(s,5,3)
imshow(x)
show()

h = σ.(W*s + b)
s = σ.(transpose(W)*h + c)
x = reshape(s,5,3)
imshow(x)
show()


