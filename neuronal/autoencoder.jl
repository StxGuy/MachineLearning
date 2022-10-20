using PyPlot

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
    
    # Adam
    ∂L∂Wm       :: Matrix
    ∂L∂bm       :: Matrix
    ∂L∂Wv       :: Matrix
    ∂L∂bv       :: Matrix
end             
    

# Architecture
mutable struct NeuralNetwork
    Layers      :: Vector{DenseLayer}
    loss        :: String
    optimizer   :: String
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

    return DenseLayer(x,W,b,z,act,dW,db,dW,db,dW,db)
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

#----------------#
# ADAM Optimizer #
#----------------#
function Adam(m,v,g,n)
    β1 = 0.9
    β2 = 0.999
    ε = 1E-8

    m = β1*m + (1-β1)*g
    v = β2*v + (1-β2)*(g.*g)
    
    mh = m ./ (1-β1^n)
    vh = v ./ (1-β2^n)
    
    Δ = mh ./ (sqrt.(vh) .+ ε)
    
    return m,v,-Δ
end

#---------------------------#
# BackPropagate Dense Layer #
#---------------------------#
function backDense!(∂L∂z,layer :: DenseLayer,opt,n)
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
    
    if (opt == "Grad")
        g1 = ∂L∂y
        g2 = ξ
    elseif (opt == "Adam")
        m,v,g1 = Adam(layer.∂L∂bm,layer.∂L∂bv,∂L∂y,n)
        setfield!(layer,:∂L∂bm,m)
        setfield!(layer,:∂L∂bv,v)
        
        m,v,g2 = Adam(layer.∂L∂Wm,layer.∂L∂Wv,ξ,n)
        setfield!(layer,:∂L∂Wm,m)
        setfield!(layer,:∂L∂Wv,v)        
    end
    
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
    setfield!(layer,:∂L∂Wm,z)
    setfield!(layer,:∂L∂Wv,z)

    i,j = size(layer.∂L∂b)
    z = zeros(i,j)
    setfield!(layer,:∂L∂b,z)
    setfield!(layer,:∂L∂bm,z)
    setfield!(layer,:∂L∂bv,z)    

    return nothing
end

#==================#
#   Architecture   #
#==================#
# Initialize Neural Network
function SetNeuralNetwork(ni,no,loss,opt)
    Layer1 = Dense(ni,4,"tanh")
    Layer2 = Dense(4,8,"tanh")
    Layer3 = Dense(8,4,"tanh")
    Layer4 = Dense(4,no,"tanh")
    
    Layers = [Layer1, Layer2, Layer3, Layer4]

    return NeuralNetwork(Layers,loss,opt)
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
function BackPropagate(∂L, nn :: NeuralNetwork,step)
    D = ∂L
    for L in nn.Layers[end:-1:1]
        D = backDense!(D,L,nn.optimizer,step)
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
function Loss(y, nn :: NeuralNetwork)
    L = nn.Layers[end]
    ŷ = L.z
    ωo² = 1
    λ = 0.8
    
    if (nn.loss == "L2")
        lossm = sum((ŷ-y).^2)/length(y)
        Dm = 2*transpose(ŷ-y)/length(y)
        
        ε_true = 0.5*y[2]^2 - ωo²*cos(y[1])
        ε_predicted = 0.5*ŷ[2]^2 -ωo²*cos(ŷ[1])
        loss_physics = (ε_predicted - ε_true)^2
        Dp = 2(ε_predicted - ε_true)*reshape([ωo²*sin(ŷ[1]) ŷ[2]],1,2)
    end
    
    loss = (1-λ)*lossm + λ*loss_physics
    D = (1-λ)*Dm + λ*Dp
    
    #println("::",lossm," ",loss_physics)
    
    return loss,D
end

# Train a neural network
function TrainNeuralNetwork!(nn :: NeuralNetwork,sample,η)
    ny,nx = size(sample)
    
    loss_y = []
    loss = 1.0
    it = 1
    while(loss > 1E-5)
        loss = 0
        for epoch in 1:ny
            # Generate data
            r = rand(1:ny)
            x = reshape(sample[r,1:2],2,1) 
            y = reshape(sample[r,3:4],2,1)
            
            # Feedforward
            ŷ = FeedForward!(x,nn)

            # Compute loss and its gradient
            l,D = Loss(y,nn)
            loss += l
            
            # Backpropagate
            D1 = BackPropagate(D,nn,it)
        end
        it += 1
        loss /= ny
        push!(loss_y,loss)
        println(loss)

        # Apply gradient descend
        SetGradients!(η, nn)
    end
    
    return loss_y
end

#====================================================#
#                       MAIN                         #
#====================================================#
function f(r)
    ωo² = 1
    θ = r[1]
    φ = r[2]
    return [φ,-ωo²*sin(θ)]
end

# Setup autoencoder
nn = SetNeuralNetwork(2,2,"L2","Grad")

# Create training set
training_set = zeros(1000,4)
Δ = 0.1

ac = 0
#for zo in [2π/3,π/2,π/4]
x = []
y = []
for zo in [π/2]# 2π/3 π/4]
    z = [zo,0]
    
    for it in 1:1000
        global ac += 1
        
        # Runge-Kutta
        training_set[ac,1] = z[1]/π + (rand()-0.5)/100
        training_set[ac,2] = z[2]/π + (rand()-0.5)/100

        k1 = f(z)
        k2 = f(z + 0.5*Δ*k1)
        k3 = f(z + 0.5*Δ*k2)
        k4 = f(z + Δ*k3)

        z += (k1 + 2k2 + 2k3 + k4)*Δ/6
        
        training_set[ac,3] = z[1]/π
        training_set[ac,4] = z[2]/π
        
        push!(x,z[1]/π)
        push!(y,z[2]/π)
    end
end
plot(x,y)

# Train autoencoder
l = TrainNeuralNetwork!(nn,training_set,0.0001)

# Print results
z = [0.5,0]
x = []
y = []
for it in 1:10000
    global z = FeedForward!(reshape(z,2,1),nn)
    push!(x,z[1])
    push!(y,z[2])
end

#plot(x)
#plot(y)
plot(x,y,color="orange")
show()

