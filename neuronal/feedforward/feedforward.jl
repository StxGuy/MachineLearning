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
end             
    

# Architecture
mutable struct NeuralNetwork
    Layers  :: Vector{DenseLayer}
    loss    :: String
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

    ∂L∂y = ∂L∂z.*D
    setfield!(layer,:∂L∂b,layer.∂L∂b + ∂L∂y)
    setfield!(layer,:∂L∂W,layer.∂L∂W + layer.x*∂L∂y)
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
function SetNeuralNetwork(ni,no,act,loss)
    nh = floor(Int,(ni+no)/2)+1

    Layer1 = Dense(ni,nh,act)
    Layer2 = Dense(nh,no,act)
    
    Layers = [Layer1, Layer2]

    return NeuralNetwork(Layers,loss)
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
function Loss(y, nn :: NeuralNetwork)
    L = nn.Layers[end]
    ŷ = L.z
    
    if (nn.loss == "L2")
        loss = sum((ŷ-y).^2)/length(y)
        D = 2*transpose(ŷ-y)/length(y)
    end
    
    return loss,D
end

# Train a neural network
function TrainNeuralNetwork!(nn :: NeuralNetwork,sample,η)
    ny,nx = size(sample)
    
    loss_y = []
    loss = 1.0
    while(loss > 1E-4)
        loss = 0
        for epoch in 1:ny
            # Generate data
            r = rand(1:4)
            x = reshape(sample[r,1:2],2,1) + 0.01*(rand(2) .- 0.5)
            y = reshape([sample[r,3]],1,1)

            # Feedforward
            ŷ = FeedForward!(x,nn)

            # Compute loss and its gradient
            l,D = Loss(y,nn)
            loss += l
            
            # Backpropagate
            D1 = BackPropagate(D,nn)
        end
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
nn = SetNeuralNetwork(2,1,"sig","L2")

training_set = [ 0 0 0;
                 0 1 1;
                 1 0 1;
                 1 1 0]

l = TrainNeuralNetwork!(nn,training_set,0.01)

# Print results
for i in 1:4
    x = reshape(training_set[i,1:2],2,1)
    ŷ = FeedForward!(x,nn)
    println(x,": ",ŷ)
end

plot(l)
show()

