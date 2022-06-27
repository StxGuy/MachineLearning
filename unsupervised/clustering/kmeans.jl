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

function kmeans(R)
    N,M = size(R)
    
    C = [0 10;4 1;11 8]
    
    for it in 1:10
        # Assignment
        for i in 1:N
            smin = 1E5
            for k in 1:3
                s = sum((R[i,1:2] - C[k,:]).^2)
                if (s < smin)
                    smin = s
                    R[i,3] = k
                end
            end
        end
        
        # Recalculate
        C = zeros(3,2)
        num = zeros(3)
        for i in 1:N
            p = trunc(Int,R[i,3])
            C[p,:] = C[p,:] + R[i,1:2]
            num[p] = num[p] + 1
        end
        
        for i in 1:3
            C[i,:] = C[i,:]/num[i]
        end
    end
    
    return R
end        


R = createdata(50)
R = kmeans(R)
scatter(R[:,1],R[:,2],10,R[:,3])
show()
