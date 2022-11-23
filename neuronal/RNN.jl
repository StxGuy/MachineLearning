using LinearAlgebra
using PyPlot

#---------------------------#
#           MODEL           #
#---------------------------#
# Dense Layer
mutable struct ElmanLayer
    Wh          :: Matrix
    Uh          :: Matrix
    bh          :: Matrix
        
    Wy          :: Matrix
    by          :: Matrix
    
    act         :: String
    
    is          :: Int
    os          :: Int
    hs          :: Int
end             
    

# Architecture
mutable struct NeuralNetwork
    Layers  :: Vector{ElmanLayer}
    loss    :: String
end

#-------------------------#
# Dense Layer Initializer #
#-------------------------#
function Elman(in_size,out_size,act)
    h_size = (in_size + out_size)÷2
    
    Wh = (rand(h_size,in_size).-0.5)./h_size
    Uh = (rand(h_size,h_size).-0.5)./h_size
    bh = zeros(h_size,1)
    
    Wy = (rand(out_size,h_size).-0.5)./out_size
    by = zeros(out_size,1)
    
    x  = zeros(in_size,1)
    h1 = zeros(h_size,1)
    
    return ElmanLayer(Wh,Uh,bh,Wy,by,act,in_size,out_size,h_size)
end

#-----------------------------------#
# Activation/deActivation Functions #
#-----------------------------------#
function sigmoid(x)
    return 1.0/(1.0 + exp(-x))
end

# Activate #
function σ(y,act)
    if (act == "tanh")
        z = tanh.(y)
    end
    if (act == "sig")
        z = sigmoid.(y)
    end
        
    return z
end

# Softmax #
function softmax(y)
    ymax = maximum(y)
    s = sum(exp.(y.-ymax))
    z = exp.(y.-ymax)/s
    
    return z
end

# Deactivate #
function dσ(z,act)
    if (act == "tanh")
        D = 1.0 .- z.^2
    end
    if (act == "sig")
        D = z .* (1.0 .- z)
    end
    
    return Diagonal(D[:,1])
end

#-------------------------#
# FeedForward Dense Layer #
#-------------------------#
function forwardElman!(x,h1,layer :: ElmanLayer)
    k = layer.Wh*x + layer.Uh*h1 + layer.bh
    h = σ(k,layer.act)
    
    y = layer.Wy*h + layer.by
    
    return h,k,y
end

function decode(x)
    p = argmax(x)
    
    if (p[1] == 4)
        s = "F"
    elseif (p[1] == 3)
        s = "="
    elseif (p[1] == 2)
        s = "m"
    else
        s = "a"
    end
end    

function feedForward(input_vector,layer :: ElmanLayer,print)
    z_vector = []
    k_vector = []
    x_vector = []
    h = zeros(layer.hs,1)
    for row in eachrow(input_vector)
        x = convert(Matrix,reshape(row,layer.is,1))
        h,k,y = forwardElman!(x,h,layer)
        z = softmax(y)
        
        push!(x_vector,x)
        push!(z_vector,z)
        push!(k_vector,k)
        if (print)
            println(decode(x)," -> ",decode(z))
        end
    end
    
    return z_vector,k_vector,x_vector
end

function BPTT(ẑv,zv,kv,xv,layer :: ElmanLayer)
    τ = length(ẑv)
    
    ∂L∂by = zeros(1,layer.os)
    ∂L∂Wy = zeros(layer.hs,layer.os)
    ∂L∂h = zeros(1,layer.hs)
    ∂L∂bh = zeros(1,layer.hs)
    ∂L∂Wh = zeros(layer.hs,layer.is)
    ∂L∂Uh = zeros(layer.hs,layer.hs)
    
    for t in τ:-1:1
        ∂L∂y = reshape(ẑv[t]-zv[t,:],1,layer.os)
        
        # y-layer
        ∂L∂by += ∂L∂y
        ∂L∂Wy += σ(kv[t],layer.act)*∂L∂y
        
        # h-layer
        if (t < τ)
            k1 = kv[t+1]
        else
            k1 = zeros(layer.hs,1)
        end
        
        dς = dσ(k1,layer.act)
        ∂L∂h = ∂L∂h*dς*layer.Uh + ∂L∂y*layer.Wy
        ∂L∂k = ∂L∂h*dς        
        ∂L∂bh += ∂L∂k
        ∂L∂Wh += xv[t]*∂L∂k
        if (t > 1)
            ∂L∂Uh += σ(kv[t-1],layer.act)*∂L∂k
        end
    end
    
    η = 0.0001
    
    setfield!(layer, :Wh, layer.Wh - η*∂L∂Wh'/τ)
    setfield!(layer, :Uh, layer.Uh - η*∂L∂Uh'/τ)
    setfield!(layer, :bh, layer.bh - η*∂L∂bh'/τ)
    setfield!(layer, :Wy, layer.Wy - η*∂L∂Wy'/τ)
    setfield!(layer, :by, layer.by - η*∂L∂by'/τ)
end

# Loss
function loss(ẑv,zv)
    τ = length(ẑv)
    
    L = 0
    for t in 1:τ
        L += sum((ẑv[t]-zv[t,:]).^2)
    end
    
    return L/τ
end


RNN = Elman(4,4,"tanh")

x = [0 0 0 1;0 0 1 0; 0 1 0 0; 1 0 0 0]
zv = [0 0 1 0;0 1 0 0; 1 0 0 0; 0 0 0 1]

ε = 1
while(ε > 0.6)
    ẑv,kv,xv = feedForward(x,RNN,false)
    global ε = loss(ẑv,zv)
    println(ε)
    BPTT(ẑv,zv,kv,xv,RNN)
end
feedForward(x,RNN,true)


