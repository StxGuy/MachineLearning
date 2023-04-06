using Flux
using PyPlot
using IterTools: ncycle

# Neural network architecture
function SetNet(inputChannels,outputChannels)
    m = Chain(
        Dense(inputChannels,32,tanh; init = Flux.glorot_normal()),
        Dense(32,outputChannels,sigmoid; init = Flux.glorot_normal())
        )
    
    return m
end

# An approximator for the neural network with an optimiser
mutable struct NNApprox
    model       :: Chain
    optimiser   :: Flux.Optimiser
end

# Returns an approximator
function SetAppr(η)
    model = SetNet(1,4)
    op = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(η))
   
   return NNApprox(model,op)
end

# Creates a dataset
function buildDataSet(N)
    x_set = Float64[0;0]
    y_set = Float64[0]
    
    for it in 1:N
        for (x1,x2,y) in zip([0 0 1 1],[0 1 0 1],[0 1 1 0])
            r1 = (rand()-0.5)/10
            r2 = (rand()-0.5)/10
            x_set = hcat(x_set,[x1+r1;x2+r2])
            y_set = hcat(y_set,y)
        end
    end
    
    return x_set,y_set
end

# Train the approximator with a dataset
# function train!(ap :: NNApprox, x :: Matrix{Float64}, y :: Matrix{Float64}, batch_size, epochs)
#     train_data = Flux.Data.DataLoader((x,y),batchsize=batch_size,shuffle=true)
#     train_data = ncycle(train_data,epochs)
#     
#     for (a,b) in train_data
#         ∇ = Flux.gradient(params(ap.model)) do
#             loss = Flux.Losses.mse(ap.model(a),b)
#         end
#         Flux.Optimise.update!(ap.optimiser,Flux.params(ap.model),∇)
#     end
# end

# Evaluate the trained approximator
function evaluate(ap :: NNApprox, M :: Matrix{Float64})
    
    for x in eachcol(M)
        ŷ = ap.model(x)
        
        println(x," = ",ŷ)
    end
end    

#================================================================================#
# REPLAY BUFFER
#================================================================================#
mutable struct ReplayBuffer
    buffer::Vector{Tuple}
    max_size::Int
    curr_size::Int
end

function ReplayBuffer(buffer_size::Int) 
    return ReplayBuffer(Vector{Tuple}(), buffer_size, 0)
end

function store!(buffer::ReplayBuffer, item) 
    if (buffer.curr_size ≥ buffer.max_size)
        popfirst!(buffer.buffer)
    else
        buffer.curr_size += 1
    end
    push!(buffer.buffer, item)
end

function sample(buffer::ReplayBuffer, batch_size::Int) 
    indices = rand(1:length(buffer.buffer), min(batch_size,buffer.curr_size))
    return [buffer.buffer[i] for i in indices]
end

#----------------------------
# REINFORCEMENT LEARNING
#----------------------------
function ε_greedy(s,ε,Q)
    sz = length(Q)
    
    if (rand() < ε)      # Exploration
        a = rand(1:sz)
    else            # Exploitation
        a = argmax(Q)
    end
    
    return a
end

function train!(N::NNApprox,R::ReplayBuffer)
    γ = 0.95
    S = sample(R,100)
    
    for (s,a,r,s′) in S
        y = r + γ*maximum(N.model([s′]))
        
        ∇ = Flux.gradient(params(N.model)) do
            loss = (y - N.model([s])[a])^2
        end
        Flux.Optimise.update!(N.optimiser,Flux.params(N.model),∇)
    end
end

# Reward matrix
rewardMatrix = 
            [-1.0 -1.0 -1.0  0.0;
             -1.0  0.5 0.0 -1.0;
             -1.0 -1.0 -1.0  0.0;
              0.0 -1.0  10.0 -1.0]

# Next state matrix      
stateMatrix = [1 1 1 2;
               2 4 1 2;
               3 3 3 4;
               2 4 3 4]

# Replay buffer
R = ReplayBuffer(10)

# Approximator for Q-value
Actor = SetAppr(1E-3)

γ = 0.95
α = 0.1
ζ = 1       # Exploration rate

y_plot = []
for t in 1:30000
    # Begin
    s = rand([1,2,4])
    totalReward = 0
    
    t = 0
    while(s ≠ 3 && t < 10)
        Q = Actor.model([s])
        a = ε_greedy(s,ζ,Q)
        r = rewardMatrix[s,a]
        totalReward += r
        s′ = stateMatrix[s,a]

        store!(R,(s,a,r,s′))
        #Q[s,a] = (1-α)*Q[s,a] + α*(r+γ*maximum(Q[s′,:]))
        s = s′
        t += 1
        
        # Reduce exploration
        global ζ = max(0.0,ζ-1E-5)
    end

    push!(y_plot,totalReward)
    println(totalReward,", ",ζ,", ",t)
    train!(Actor,R)
end

function showmat(M)
    for m in eachrow(M)
        println(m)
    end
end

#showmat(Q)
for s in 1:4
    println(Actor.model([s]))
end

plot(y_plot)
show()

#y = r + γ*maximum(Q(s′))
#loss = mse(y,Q)


#================================================================#
# MAIN CODE
#================================================================#
# Q = SetAppr(1E-3)
# 
# a = evaluate(Q,s′,a′)
# 
# x,y = buildDataSet(200)
# train!(A,x,y,20,100)
# 
# x = Float64[0 0 1 1;0 1 0 1]
# evaluate(A,x)
