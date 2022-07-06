using PyPlot


η = 0.1

# Euclidean distance
function d(x,y)
    m,n = size(x)
    z = []

    for i in 1:n
        push!(z,(x[1,i]-y[1])^2 + (x[2,i]-y[2])^2)
    end

    return z
end


# Original data
x_sp = LinRange(0,2π,100)
y_sp = sin.(x_sp)
x_sp = hcat(x_sp,y_sp)'

# Initial guess
w = zeros(2,10)
for i in 1:10
    w[1,i] = rand(x_sp[1,:])
    w[2,i] = rand()-0.5
end

# LVQ Loop
for it in 1:10000
    r = rand(1:100)
    x = x_sp[:,r]                       # Choose input vector randomly

    n = argmin(d(w,x))                  # Find closest output
    w[:,n] = w[:,n] + η*(x-w[:,n])      # Move winning node towards input
end

plot(x_sp[1,:],x_sp[2,:])
scatter(w[1,:],w[2,:])
show()
