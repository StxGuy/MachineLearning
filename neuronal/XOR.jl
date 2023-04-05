using Flux
using IterTools: ncycle

# Neural network architecture
function SetNet(inputChannels,outputChannels)
    m = Chain(
        Dense(2,32,tanh; init = Flux.glorot_normal()),
        Dense(32,1,sigmoid; init = Flux.glorot_normal())
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
    model = SetNet(2,1)
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
function train!(ap :: NNApprox, x :: Matrix{Float64}, y :: Matrix{Float64}, batch_size, epochs)
    train_data = Flux.Data.DataLoader((x,y),batchsize=batch_size,shuffle=true)
    train_data = ncycle(train_data,epochs)
    
    for (a,b) in train_data
        ∇ = Flux.gradient(params(ap.model)) do
            loss = Flux.Losses.mse(ap.model(a),b)
        end
        Flux.Optimise.update!(ap.optimiser,Flux.params(ap.model),∇)
    end
end

# Evaluate the trained approximator
function evaluate(ap :: NNApprox, M :: Matrix{Float64})
    for x in eachcol(M)
        ŷ = ap.model(x)
        
        println(x," = ",ŷ)
    end
end    
    
#================================================================#
# MAIN CODE
#================================================================#
A = SetAppr(1E-3)

x,y = buildDataSet(200)
train!(A,x,y,20,100)

x = Float64[0 0 1 1;0 1 0 1]
evaluate(A,x)
