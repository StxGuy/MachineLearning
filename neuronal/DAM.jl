using LinearAlgebra
using PyPlot

p0 = [1 1 1;
      1 0 1;
      1 0 1;
      1 0 1;
      1 1 1]
p0 = reshape(2*(p0 .- 0.5),15,1)
      
p1 = [0 0 1;
      0 1 1;
      0 0 1;
      0 0 1;
      0 0 1]
p1 = reshape(2*(p1 .- 0.5),15,1)      
      
p2 = [1 1 1;
      0 0 1;
      0 1 0;
      1 0 0;
      1 1 1]
p2 = reshape(2*(p2 .- 0.5),15,1)
      
p3 = [1 1 1;
      0 0 1;
      0 1 1;
      0 0 1;
      1 1 1]
p3 = reshape(2*(p3 .- 0.5),15,1)
      
p4 = [1 0 1;
      1 0 1;
      1 1 1;
      0 0 1;
      0 0 1]
p4 = reshape(2*(p4 .- 0.5),15,1)
      
p5 = [1 1 1;
      1 0 0;
      1 1 1;
      0 0 1;
      1 1 1]
p5 = reshape(2*(p5 .- 0.5),15,1)
      
p6 = [1 0 0;
      1 0 0;
      1 1 1;
      1 0 1;
      1 1 1]
p6 = reshape(2*(p6 .- 0.5),15,1)
      
p7 = [1 1 1;
      0 0 1;
      0 1 0;
      0 1 0;
      0 1 0]
p7 = reshape(2*(p7 .- 0.5),15,1)
      
p8 = [1 1 1;
      1 0 1;
      0 1 0;
      1 0 1;
      1 1 1]
p8 = reshape(2*(p8 .- 0.5),15,1)
      
p9 = [1 1 1;
      1 0 1;
      1 1 1;
      0 0 1
      0 0 1]      
p9 = reshape(2*(p9 .- 0.5),15,1)

W = transpose([p0 p1 p2 p3 p4 p5 p6 p7 p8 p9])

function sgn(x)
    if (x > 0)
        return 1
    else
        return -1
    end
end

function softmax(x)
    m = maximum(x)
    
    S = 0
    for xi in x
        S += exp(xi-m)
    end
    
    return exp.(x .- m)/S
end


# Output
s = [1 1 0;
     0 0 1;
     0 1 0;
     1 1 0;
     0 1 1]
s = reshape(2*(s .- 0.5),15,1)     

β = 0.9
for it in 1:8
    x = reshape(s,5,3)
    imshow(x)
    show()
    
    h = β*W*s
    global s = (transpose(W)*softmax(h))
end


