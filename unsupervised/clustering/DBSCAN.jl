using PyPlot

function createdata(N)
    R = []
    for i in 1:N
        r = 5+rand()*2
        θ = (rand()-0.5)*π
        x = r*cos(θ)
        y = r*sin(θ)
    
        if (i == 1)
            R = [x y 0]
        else
            R = vcat(R,[x y 0])
        end
        
        x = randn()
        y = randn()
        R = vcat(R,[x y 0])        
    end
    
    return R
end

function Euclidean(x,y)
    return sum((x-y).^2)
end

function neighborhood(p,R,dmin)
    N,M = size(R)
    nei = []
    
    for i in 1:N
        d = Euclidean(R[p,1:2],R[i,1:2])
        if (d < dmin)
            push!(nei,i)
        end
    end
    
    return nei
end

function clusterize(p,R,cluster,Nmin,dmin)
    list = []
    push!(list,p)
    
    while(length(list) != 0)
        q = pop!(list)
        
        nei = neighborhood(q,R,dmin)
        if (R[q,3] == 0 && length(nei) > Nmin)
            R[q,3] = cluster
            while(length(nei) > 0)
                t = pop!(nei)
                push!(list,t)
            end
        end
    end
    
    return R
end    

function Scan(R,Nmin,dmin)
    N,M = size(R)
    
    C = 0
    for i in 1:N
        nei = neighborhood(i,R,dmin)
        if (length(nei) > Nmin && R[i,3] == 0)
            C = C + 1
            R = clusterize(i,R,C,Nmin,dmin)
        end
    end
    
    return R
end

R = createdata(150)
R = Scan(R,3,1)
scatter(R[:,1],R[:,2],10,R[:,3])
show()
