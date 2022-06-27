function getData(fname)
    mat = zeros(252,7)
    f = open(fname,"r")
    for i in 1:252
        line = readline(f)
        columns = split(line,"\t")
        for j in 1:7
            mat[i,j] = parse(Float64,columns[j])
        end
    end
    close(f)
    
    return mat
end


function Euclidean(M,i,j)
    t, = size(M)
    return sum((M[:,i]-M[:,j]).^2)/t
end

function Manhattan(M,i,j)
    return sum(abs.(M[:,i]-M[:,j]))
end

function Chebyshev(M,i,j)
    return maximum(M[:,i]-M[:,j])
end

function showmat(X)
    i,j = size(X)
    
    for k in 1:i
        println(X[k,:])
    end
end 

function AGNES(M)
    t,NClu = size(M)
    c1 = 0
    c2 = 0
    
    Labels = ["INTC","AMD","IBM","NVDA","MCHP","ADI","TXN"]

    # Create distance matrix
    D = zeros(7,7)
    for i in 1:(NClu-1)
        for j in (i+1):NClu
            d = Euclidean(M,i,j)
            D[i,j] = d
            D[j,i] = D[i,j]
        end
    end
    
    # Clustering loop
    for it in 1:6
        d_min = 1E7
        for i in 1:(NClu-1)
            for j in (i+1):NClu
                if (D[i,j] < d_min)
                    d_min = D[i,j]
                    c1 = i
                    c2 = j
                end
            end
        end
    
        # Create mapping
        map = zeros(Int,NClu)
        j = 1
        for i in 1:NClu
            if (i != c1 && i != c2)
                map[i] = j
                j = j + 1
            else
                map[i] = 0
            end
        end
        
        # New distance matrix
        nL = ["" for i in 1:NClu-1]
        nD = zeros(NClu-1,NClu-1)
        for j in 1:NClu
            if (j != c1 && j != c2)
                nL[map[j]] = Labels[j]
                for i in 1:NClu
                    if (i != c1 && i != c2)
                        nD[map[i],map[j]] = D[i,j]
                    end
                end
            end
        end
        
        # Add new distances
        for i in 1:NClu
            if (i != c1 && i != c2)
                nL[NClu-1] = Labels[c1]*"/"*Labels[c2]
                d = 0.5*(D[i,c1]+D[i,c2])
                nD[map[i],NClu-1] = d
                nD[NClu-1,map[i]] = d
            end
        end
                        
        println("Link: ",Labels[c1]," and ",Labels[c2]," with distance ",d_min,", forming new cluster ",nL[NClu-1])
        println("-------------------------------------------------------------------")
        
        NClu = NClu - 1
        Labels = nL
        D = nD
        
        #showmat(D)
    end
end

M = getData("data.dat")
AGNES(M)
