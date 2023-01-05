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

function softmax(v)
    ma = maximum(v)
    
    n = exp.(v .- ma)
    
    return n/sum(n)
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
    if (layer.activation == "none")
        z = copy(y)
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
        ∂L∂y = ∂L∂z.*D
    end
    if (layer.activation == "sig")
        D = (layer.z .* (1.0 .- layer.z))'
        ∂L∂y = ∂L∂z.*D
    end
    if (layer.activation == "none")
        ∂L∂y = copy(∂L∂z)
    end
    
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



#----------------------------------------------------------#
# Initialize Neural Network
function SetNeuralNetwork()
    Layer1 = Dense(4,120,"tanh")
    Layer2 = Dense(120,4,"none")
    
    Layers = [Layer1, Layer2]
    
    return NeuralNetwork(Layers)
end

function roulette_selection(fitness)
    total_fitness = sum(fitness)
    r = rand()*total_fitness
    
    i = 0
    while(r ≥ 0)
        i += 1
        r -= fitness[i]
    end
    
    return i
end

function embedding(s)
    if (s == 1)
        state = reshape([1,0,0,0],4,1)
        elseif (s == 2)
        state = reshape([0,1,0,0],4,1)
        elseif (s == 3)
        state = reshape([0,0,1,0],4,1)
    else
        state = reshape([0,0,0,1],4,1)
    end    
    
    return state
end

# Train a neural network
function TrainNeuralNetwork!(nn :: NeuralNetwork,RewardMatrix,StateTransitionMatrix)
    η = 1E-6
    num_episodes = 100
    
    loss = []
    for epoch in 1:10000
        avR = 0
        avS = 0
        for episode in 1:num_episodes
            s = rand([1,2,4])
            ∇ = zeros(1,4)
            R = 0
            steps = 0
            while(s ≠ 3 && steps < 5)
                steps += 1
                neural_output = FeedForward!(embedding(s),nn)
                action_probabilities = softmax(neural_output)
                action = roulette_selection(action_probabilities)
                            
                R += RewardMatrix[s,action]
                
                z = zeros(1,4)
                z[1,action] = 1
                ∇ -= z - reshape(action_probabilities,1,4)
                
                s = StateTransitionMatrix[s,action]
            end

            ∇ *= R
            BackPropagate(∇,nn)
            
            avR += R
            avS += steps        
        end 
        
        avR /= num_episodes
        avS /= num_episodes
        
        SetGradients!(η,nn)
        push!(loss,avS)
        println("Epoch ",epoch,", average reward: ",avR,", average steps: ",avS)
    end
        
    
    return loss
end

function CheckPolicy(nn :: NeuralNetwork)
    for s in [1,2,4]
        neural_output = FeedForward!(embedding(s),nn)
        action_probabilities = softmax(neural_output)
        println(s,": ",action_probabilities)
    end
end

#====================================================#
#                       MAIN                         #
#====================================================#
# States
# A = [1 0 0 0]
# B = [0 1 0 0]
# C = [0 0 1 0]
# D = [0 0 0 1]

# Actions
# up    = [1 0 0 0]
# down  = [0 1 0 0]
# left  = [0 0 1 0]
# right = [0 0 0 1]

nn = SetNeuralNetwork()

# Reward matrix
Rm = [-1 -1  -1  0;
      -1  0   0 -1;
      -1 -1  -1  0;
       0 -1 100 -1]

# Next state matrix      
M = [1 1 1 2;
     2 4 1 2;
     3 3 3 4;
     2 4 3 4]

R = TrainNeuralNetwork!(nn,Rm,M)
CheckPolicy(nn)
plot(R)
show()


