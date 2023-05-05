using PyPlot

mutable struct simplex
    x1  :: Vector
    x2  :: Vector
    x3  :: Vector
    
    f1  :: Float64
    f2  :: Float64
    f3  :: Float64
       
    α   :: Float64
    γ   :: Float64
    ρ   :: Float64
    β   :: Float64
end

function simplex(x1 :: Vector, x2 :: Vector, x3 :: Vector)
    return simplex(x1,x2,x3,
                   0.0,    # f1
                   0.0,    # f2
                   0.0,    # f3
                   1.0,    # α
                   2.0,    # γ
                   0.5,    # ρ
                   0.5)    # β
end

function Rosen(r :: Vector)
    a = 1.0
    b = 100.0
    
    return (a-r[1])^2 + b*(r[2]-r[1]^2)^2
end

function order!(s :: simplex)
    f1 = Rosen(s.x1)
    f2 = Rosen(s.x2)
    f3 = Rosen(s.x3)
    
    f = [f1,f2,f3]
    x = [s.x1,s.x2,s.x3]
    
    index = sortperm(f)
    
    setfield!(s,:x1,x[index[1]])
    setfield!(s,:x2,x[index[2]])
    setfield!(s,:x3,x[index[3]])
    
    setfield!(s,:f1,f[index[1]])
    setfield!(s,:f2,f[index[2]])
    setfield!(s,:f3,f[index[3]])
end

function step!(s :: simplex)
    # Sort 
    order!(s)
    
    # Find centroid
    xo = (s.x1 + s.x2)/2
    
    # Find reflection
    xr = xo + s.α*(xo - s.x3)
    fr = Rosen(xr)
        
    # Reflected better than best solution?
    if (fr < s.f1)
        # Try to expand
        xe = xo + s.γ*(xr - xo)
        fe = Rosen(xe)
        
        if (fr ≤ fe)    # Reflected is still better
            setfield!(s,:x3,xr)
            setfield!(s,:f3,fr)
        else
            setfield!(s,:x3,xe)
            setfield!(s,:f3,fe)
        end
    elseif (s.f1 ≤ fr < s.f2)   # Reflected worse than all but worst?
        setfield!(s,:x3,xr)
        setfield!(s,:f3,fr)
    else
        # Find contraction
        if (fr < s.f3)
            xc = xo + s.ρ*(xr - xo)
        else
            xc = xo + s.ρ*(s.x3 - xo)
        end
        fc = Rosen(xc)
            
        if (fc < s.f3)
            setfield!(s,:x3,xc)
            setfield!(s,:f3,fc)
        else
            shrink!(s)
        end
    end
    
    f1 = Rosen(s.x1)
    f3 = Rosen(s.x3)
    
    return f3#abs(f1-f3)/(f1+f3)
end

function shrink!(s :: simplex)
    x2 = s.x1 + s.β*(s.x2 - s.x1)
    x3 = s.x1 + s.β*(s.x3 - s.x1)
    
    setfield!(s,:x2, x2)
    setfield!(s,:x3, x3)
end

function main()
    x1 = [-0.532,1.012]
    x2 = [1.575,-1.565]
    x3 = [1.212,2.532]
    s = simplex(x1,x2,x3)
    
    x_space = LinRange(-2,2,300)
    y_space = LinRange(-1,3,300)
    R = zeros(300,300)
    for j in 1:300
        for i in 1:300
            R[j,i] = log(Rosen([x_space[i],y_space[j]]))
        end
    end
    #contourf(x_space,y_space,R,cmap="gray")
    
    ε = []
    cnt = 0
    while(true)
        cnt += 1
        if (step!(s) < 1E-6)
            break
        end
        
        push!(ε,s.f3)
        
        if (mod(cnt-10,5) == 0)
            xx = [s.x1[1],s.x2[1],s.x3[1],s.x1[1]]
            yy = [s.x1[2],s.x2[2],s.x3[2],s.x1[2]]
            #plot(xx,yy,color="white")
        end
    end
    
    println(s.x1[1]," ",s.x1[2])
    

    semilogy(ε,color="darkgray")
    #axis([-1.0,1.5,-1.0,2.0])
    axis([0,66,1E-6,1E2])
    xlabel("Step")
    ylabel("Log(error)")
    show()
end

main()
    
            
        
    
    

    
