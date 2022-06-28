using PyPlot

function createdata(N)
    R = []
    for i in 1:N
        for (kx,ky) in zip([1 5 10],[10 1 7])
            x = kx + randn()
            y = ky + randn()
            z = rand([1 2 3])
        
            if (i == 1)
                R = [x y z]
            else
                R = vcat(R,[x y z])
            end
        end
    end
    
    return R
end

# Average intracluster distance
function Cost(M,R)
    N,M = size(R)
    
    s = zeros(3)
    n = zeros(3)
    for i in 1:(N-1)
        p = trunc(Int,R[i,3])
        for j in (i+1):N
            q = trunc(Int,R[j,3])
            if (p == q)
                s[p] = s[p] + Euclidean(R[i,1:2],R[j,1:2])
                n[p] = n[p] + 1
            end
        end
    end
    
    return sum(s./n)
end
                
# Partition around medoids
function PAM(R)
    N,M = size(R)
    
    # Initial conditions
    Medoids = [0 0 0]
    while(Medoids[1] == 0 || Medoids[2] == 0 || Medoids[3] == 0)
        i = rand(1:N)
        p = trunc(Int,R[i,3])
        Medoids[p] = i
    end
        
    for it in 1:50
        # Assign
        for i in 1:N
            dmin = 1E5
            for k in 1:3
                d = Euclidean(R[i,1:2],R[Medoids[k],1:2])
                if (d < dmin)
                    dmin = d
                    R[i,3]= k
                end
            end
        end
        
        # Recalculate
        BestCost = Cost(Medoids,R)
        for i in 1:N
            MedoidsBK = Medoids
            p = trunc(Int,R[i,3])
            Medoids[p] = i
            c = Cost(Medoids,R)
            if (c < BestCost)
                BestCost = c
            else
                Medoids = MedoidsBK
            end
        end
    end
        
    return R
end     

function Euclidean(x,y)
    t = length(x)
    return sum((x-y).^2)/t
end

function silhouette(R)
    N,M = size(R)
    NC = 3
    s = zeros(N)
    
    # Find size of clusters
    nC = zeros(NC)
    for i in 1:N
        p = trunc(Int,R[i,3])
        nC[p] = nC[p] + 1
    end
    
    # Scan other elements
    for i in 1:N
        t = zeros(NC)
        p = trunc(Int,R[i,3])
        a = 0
        for j in 1:N
            q = trunc(Int,R[j,3])
            d = Euclidean(R[i,1:2],R[j,1:2])
            if (p == q)
                a = a + d/(nC[q]-1)
            else
                t[q] = t[q] + d/nC[q]
            end
        end        
        b = minimum(t[1:NC .≠ p])
        
        # Silhouette itself
        if (a < b)
            s[i] = 1-a/b
        elseif (a > b)
            s[i] = b/a-1
        else
            s[i] = 0
        end
    end
    
    return s
end
               

R = createdata(50)
R = PAM(R)
scatter(R[:,1],R[:,2],10,R[:,3])
#plot(silhouette(R),"o")
show()
