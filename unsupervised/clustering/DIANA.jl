function getData(fname)
    mat = Array{Any}(undef,253,7)
    mat[1,:] = ["INTC","AMD","IBM","NVDA","MCHP","ADI","TXN"]
    f = open(fname,"r")
    for i in 2:253
        line = readline(f)
        columns = split(line,"\t")
        for j in 1:7
            mat[i,j] = parse(Float64,columns[j])
        end
    end
    close(f)
    
    return mat
end

function Euclidean(x,y)
    t = length(x)
    return sum((x-y).^2)/t
end


function avg(x,A)
    d = 0
    m = length(A)
        
    if (m == 0)
        return 0
    else
        for y in A
            d = d + Euclidean(x[2:end],y[2:end])
        end
        
        return d/(length(A)-1)
    end
end

function diam(A)
    dmax = -1
    for x in A
        for y in A
            d = Euclidean(x[2:end],y[2:end])
            if (d > dmax)
                dmax = d
            end
        end
    end
    
    return dmax
end            

function DIANA_split(C)
    # Initial conditions
    A = Set(C)
    B = Set()

    # Run until Dmax > 0
    L = 1
    while(L > 0)
        D_max = -1
        v = []
        for x in A
            dA = avg(x,setdiff(A,[x]))
            dB = avg(x,B)
            D = dA - dB
                    
            if (D > 0 && D > D_max)
                D_max = D
                v = x
            end
        end
        
        # If Dmax > 0, then A = A\v, B = B U v
        L = length(v)
        if (L > 0)
            setdiff!(A,[v])
            push!(B,v)
        end
    end

    # Split cluster
    return A,B
end 

function DIANA(M)
    # Initial conditions
    t,NClu = size(M)
    C = Set([M[:,i] for i in 1:NClu])
    Clusters = Set([C])
    
    for it in 1:7
        # Find largest cluster
        Lmax = -1
        csplit = []
        for c in Clusters
            L = diam(c)
            if (L > Lmax)
                Lmax = L
                csplit = c
            end
        end
        
        # If cluster has more than one element, split
        if (length(csplit) > 1)
            setdiff!(Clusters,[csplit])
            A,B = DIANA_split(csplit)
            push!(Clusters,A)
            push!(Clusters,B)
            
            # print
            println("Cluster:")
            for x in csplit
                println("  ",x[1])
            end
            println("Split into:")
            for x in A
                println("  ",x[1])
            end
            println("And:")
            for x in B
                println(" ",x[1])
            end
            println("-----------------------")
        end
    end
end

M = getData("data.dat")
DIANA(M)
