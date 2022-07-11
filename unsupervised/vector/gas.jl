using PyPlot




# Euclidean distance
function d(x,y)
    m,n = size(x)
    z = []

    for i in 1:n
        push!(z,(x[1,i]-y[1])^2 + (x[2,i]-y[2])^2)
    end

    return z
end

# Quantization error
function qerr(sub,mani)
    sm, sn = size(sub)
    mm, mn = size(mani)

    d_av = 0
    for s in 1:sn
        d_min = 1E8
        for m in 1:mn
            d = (sub[1,s]-mani[1,m])^2 + (sub[2,s]-mani[2,m])^2
            if (d < d_min)
                d_min = d
            end
        end
        d_av = d_av + d_min
    end

    return d_av/sn
end

# Connect nodes
function connect(i,j,C,H)
    n,m = size(C)

    # Connect nodes
    C[i,j] = 1
    C[j,i] = 1

    # Set history to zero
    H[i,j] = 0
    H[j,i] = 0

    # Age edges
    for k in 1:n
        if (C[i,k] == 1)
            H[i,k] = H[i,k] + 1
            H[k,i] = H[k,i] + 1

            # Remove old edges
            if (H[i,k] > 10)
                C[i,k] = 0
                C[k,i] = 0
                H[i,k] = 0
                H[k,i] = 0
            end
        end
    end

    return C,H
end

# Draw network
function draw_net(w,C)
    n,m = size(C)

    for i in 1:(n-1)
        for j in (i+1):n
            if (C[i,j] == 1)
                plot([w[1,i],w[1,j]],[w[2,i],w[2,j]])
            end
        end
    end
end

# Neural gas
function ggas(x_sp,Nr)
    # Initial guess
    w = zeros(2,Nr)
    for i in 1:Nr
        w[1,i] = rand()*2π
        w[2,i] = 2*(rand()-0.5)
    end

    # Connection and history
    C = zeros(Nr,Nr)
    H = zeros(Nr,Nr)

    β = 5
    ηo = 1E-3

    # VQ Loop
    err = 1
    cnt = 0
    while(err > 0.01 && cnt < 1E6)
        # Choose input vector randomly
        chosen = rand(1:N)
        x = x_sp[:,chosen]

        # Sort
        ki = sortperm(d(w,x))   # Indexes
        k = (ki.-1)/(Nr-1)      # Weights

        # Update topological neigborhood
        for i in 1:Nr
            if (C[ki[1],i] == 1 || ki[1] == i)
                η = ηo*exp(-β*k[i])
                w[:,i] = η*x + (1-η)*w[:,i]
            end
        end

        C,H = connect(ki[1],ki[2],C,H)
        err = qerr(w,x_sp)
        println(cnt,",",err)
        cnt = cnt + 1
    end

    return w,C
end

# Original data
N = 100
x_sp = LinRange(0,2π,N)
y_sp = sin.(x_sp)
x_sp = hcat(x_sp,y_sp)'

# Neural gas
w,C = ggas(x_sp,20)

# Draw
draw_net(w,C)
plot(x_sp[1,:],x_sp[2,:])
scatter(w[1,:],w[2,:])
show()
