using PyPlot
using LinearAlgebra

#---------------------------#
#           MODEL           #
#---------------------------#
# Dense Layer
mutable struct DenseLayer
    x           :: Matrix
    W           :: Matrix
    b           :: Matrix
    z           :: Matrix
    activation  :: String
    ∂L∂W        :: Matrix
    ∂L∂b        :: Matrix
end             

# Architecture
mutable struct NeuralNetwork
    Layers      :: Vector{DenseLayer}
end

#-------------------------#
# Dense Layer Initializer #
#-------------------------#
function Dense(input_size,output_size,act)
    x = zeros(input_size,1)
    W = (rand(output_size,input_size) .- 0.5) ./ output_size
    b = (rand(output_size,1) .- 0.5)
    z = zeros(output_size,1)
    dW = zeros(input_size,output_size)
    db = zeros(1,output_size)

    return DenseLayer(x,W,b,z,act,dW,db)
end

#----------------------#
# Activation Functions #
#----------------------#
function sigmoid(x)
    return 1.0/(1.0 + exp(-x))
end

function relu(x)
    if (x > 0)
        return x
    else
        return 0
    end
end

function Heaviside(x)
    if (x > 0)
        return 1.0
    else
        return 0.0
    end
end

#-------------------------#
# FeedForward Dense Layer #
#-------------------------#
function forwardDense!(x,layer :: DenseLayer)
    setfield!(layer,:x,x)
    y = layer.W*x + layer.b

    if (layer.activation == "tanh")
        z = tanh.(y)
    end
    if (layer.activation == "sig")
        z = sigmoid.(y)
    end
    if (layer.activation == "relu")
        z = relu.(y)
    end

    setfield!(layer,:z,z)

    return z
end

#---------------------------#
# BackPropagate Dense Layer #
#---------------------------#
function backDense!(∂L∂z,layer :: DenseLayer)
    if (layer.activation == "tanh")
        D = 1.0 .- layer.z'.^2
    end
    if (layer.activation == "sig")
        D = (layer.z .* (1.0 .- layer.z))'
    end
    if (layer.activation == "relu")
        D = Heaviside.(layer.z')
    end

    ∂L∂y = ∂L∂z.*D
    ξ = layer.x*∂L∂y
    g1 = ∂L∂y
    g2 = ξ
    
    setfield!(layer,:∂L∂b,layer.∂L∂b + g1)
    setfield!(layer,:∂L∂W,layer.∂L∂W + g2)
    
    ∂L∂x = ∂L∂y*layer.W
        
    return ∂L∂x
end

#------------------#
# Gradient Descend #
#------------------#
function Gradient!(layer :: DenseLayer, η)
    setfield!(layer,:W, layer.W - η*layer.∂L∂W')
    setfield!(layer,:b, layer.b - η*layer.∂L∂b')

    i,j = size(layer.∂L∂W)
    z = zeros(i,j)
    setfield!(layer,:∂L∂W,z)

    i,j = size(layer.∂L∂b)
    z = zeros(i,j)
    setfield!(layer,:∂L∂b,z)

    return nothing
end

#==================#
#   Architecture   #
#==================#
# Initialize Neural Network
function SetNeuralNetwork(fans,acts)
    Layers = []

    for i in 1:(length(fans)-1)
        Layer = Dense(fans[i],fans[i+1],acts[i])
        push!(Layers,Layer)
    end

    return NeuralNetwork(Layers)
end

# Feedforward Neural Network
function FeedForward!(x, nn :: NeuralNetwork)
    z = x
    for L in nn.Layers
        z = forwardDense!(z,L)
    end
        
    return z
end

# Backpropagate Neural Network
function BackPropagate(∂L, nn :: NeuralNetwork)
    D = ∂L
    for L in nn.Layers[end:-1:1]
        D = backDense!(D,L)
    end

    return D
end

# Apply gradients to Neural Network
function SetGradients!(η, nn :: NeuralNetwork)
    for L in nn.Layers
        Gradient!(L,η)
    end
    
    return nothing
end

# Loss
function L2(ŷ,y)
    loss = sum((ŷ-y).^2)/length(y)
    grad = 2*transpose(ŷ-y)/length(y)
    return loss,grad
end

# Kullback-Leibler
function Dkl(μ,Σ)
    loss = 0.5*(-log(det(Σ)) + tr(Σ) + dot(μ,μ) - length(μ))
    grad = [transpose(μ), 0.5*(1 .- inv(Σ))]
    return loss,grad
end

#====================================================#
#                       MAIN                         #
#====================================================#
# Parameters & Constants
η = 0.001
t = LinRange(0,2π,32)
error = 10
it = 0

# Setup autoencoder
sx = SetNeuralNetwork([32,8],["tanh"])
h = SetNeuralNetwork([8,1],["tanh"])
g = SetNeuralNetwork([8,1],["sig"])
f = SetNeuralNetwork([1,32],["tanh"])

# Train variational autoencoder
while(error > 1E-2)
    θ = (0.1*rand(1:10)-0.5)*π      # Finite sampling space
    x = reshape(sin.(t.+θ),32,1)

    # Feedforward
    s = FeedForward!(x,sx)
    μ = FeedForward!(s,h)
    σ² = FeedForward!(s,g)
    ξ = randn()
    z = μ + sqrt(σ²)*ξ
    xh = FeedForward!(z,f)

    # Losses
    Loss1,∂L1 = L2(xh,x)
    Loss2,∂L2 = Dkl(μ,σ²)
    global error = Loss1
    
    # Progress time
    println(Loss1," ",Loss2)
    r = it/1E8
    if (it ≤ 1E6)
        global it = it + 1
    end
    
    # Backpropagate
    ∂L∂z = BackPropagate(∂L1,f)
    ∂L∂μ = ∂L∂z + ∂L2[1]*r
    ∂L∂σ² = 0.5*∂L∂z*ξ/sqrt(σ²) + ∂L2[2]*r
    ∂L∂s1 = BackPropagate(∂L∂μ,h)
    ∂L∂s2 = BackPropagate(∂L∂σ²,g)
    ∂L∂s = BackPropagate(∂L∂s1 + ∂L∂s2,sx)

    # Apply gradients
    SetGradients!(η, sx)
    SetGradients!(η, h)
    SetGradients!(η, g)
    SetGradients!(η, f)
end

for it in 1:5
    # Make prediction
    θ = (0.1*rand(1:10)-0.5)*π
    x = reshape(sin.(t.+θ),32,1)

    s = FeedForward!(x,sx)
    μ = FeedForward!(s,h)
    σ = FeedForward!(s,g)
    ξ = randn()
    z = μ + σ*ξ
    xh = FeedForward!(z,f)

    # Plot
    plot(t/π,x)
    plot(t/π,xh)
    xlabel("θ/π")
    show()
end
