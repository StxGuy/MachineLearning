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

patterns = [p0 p1 p2]# p3 p4 p5 p6 p7 p8 p9]

function sgn(x)
    if (x > 0)
        return 1
    else
        return -1
    end
end

# Training
W = zeros(15,15)
for x in eachcol(patterns)
    global W += (x*transpose(x) - I)
end
W /= size(patterns,1)

# Output
s = [1 1 0;
     0 0 1;
     0 1 0;
     1 1 0;
     0 1 1]
s = reshape(2*(s .- 0.5),15,1)     

Energy1 = (-0.5*s'*W*s)[1]

while(Energy1 > -8)
    println(Energy1)

    x = reshape(s,5,3)
    imshow(x)
    show()
    
    p = rand(1:15)
    t = copy(s)
    t[p] = -t[p]
    Energy2 = (-0.5*t'*W*t)[1]
    
    if (Energy2 < Energy1)
        global s = copy(t)
        global Energy1 = Energy2
    end
end


