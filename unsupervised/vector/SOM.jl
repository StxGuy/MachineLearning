using PyPlot

# Euclidean distance
function distance(x,y)
    m,n = size(x)
    z = []

    for i in 1:n
        push!(z,(x[1,i]-y[1])^2 + (x[2,i]-y[2])^2)
    end

    return z
end

# Read image and create vector x
function loadData()
    data = []
    dx = []
    dy = []
    r = 2
    for it in 1:200
        θ = rand()*2π
        x = r*cos(θ)
        y = r*sin(θ)

        push!(data,[y x])
        push!(dx,x)
        push!(dy,y)
    end

    return data,dx,dy
end

#=========================================================#
x,dx,dy = loadData()
N = length(x)
println(N," vectors")
Nr = 20    # Number of new vectors

# Parameters
σo = 0.2
τσ = 1E5
τη = 1E5
ηo = 0.1

# Initial conditions
w = (rand(2,Nr).-0.5)*6
t = 0

# SOM loop
for it in 1:100000
    r = rand(1:N)           # Choose random vector
    d = distance(w,x[r])    # Find BMU
    i = argmin(d)
    BMU = w[:,i]

    # Time-dependent parameters
    σ² = 1E-10 + σo*exp(-t/τσ)
    η = ηo*exp(-t/τη)
    global t = t + 1

    # Move everybody towards BMU
    for i in 1:Nr
        d = sqrt((w[1,i]-BMU[1])^2 + (w[2,i]-BMU[2])^2)
        Nei = exp(-d/σ²)
        w[:,i] = w[:,i] + η*Nei*(x[r]'-w[:,i])
    end

    # Error
    d_max = 0
    for i in 1:Nr
        d_min = 1E8
        for j in 1:N
            d = sqrt((w[1,i]-x[j][1])^2+(w[2,i]-x[j][2])^2)
            if (d < d_min)
                d_min = d
            end
        end
        if (d_min > d_max)
            d_max = d_min
        end
    end
    println(it,": ",d_max,", η=",η,", σ²=",σ²)
end

scatter(dx,dy)
scatter(w[1,:],w[2,:])
show()
