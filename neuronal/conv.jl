using Statistics
using MLDatasets

using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser
using Flux.Losses: logitcrossentropy


# Get MNIST data and create minibatch
# Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner
# "Gradient-based learning applied to document recognition." 
# In: Proceedings of the IEEE, 86-11 (1998) 2278-2324.

function get_data(bsize = 128)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize=bsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest),  batchsize=bsize)
    
    return train_loader, test_loader
end

# Build a convolutional model
function build_model()
    model = Chain(
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        MaxPool((2,2)), 

        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)), 

        Conv((3, 3), 32=>32,pad=(1,1), relu),
        MaxPool((2,2)),

        flatten,
        Dense(288, 10),

        softmax,
    )
    
    model = gpu(model);
    
    return model
end

# Loss and accuracy functions
loss(ŷ, y) = logitcrossentropy(ŷ, y)
accuracy(x,y,model) = mean(onecold(model(x)) .== onecold(y))

# Train model and make predictions
function train()
    train_loader,test_loader = get_data()
    model = build_model()
    ps = Flux.params(model)
    opt = ADAM(0.001)
    
    # Train
    for epoch in 1:10
        for (x,y) in train_loader
            x,y = x |> gpu, y |> gpu
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ,y)
            end
            
            Flux.Optimise.update!(opt,ps,gs)
        end
        
        # Print accuracy
        ac = 0
        n = 0
        for (x,y) in train_loader
            x,y = x |> gpu, y |> gpu
            ac += accuracy(x,y,model)
            n += size(x)[end]
        end
        println(100*ac/n)            
    end
    
   return model    
end

# Make some predictions
function predict(model)
    x,y = MLDatasets.MNIST(:test)[:]
    ŷ = zeros(10)
    
    for i in 1:5
        g = reshape(x[:,:,i],28,28,1,1)
        g = g |> gpu
        copyto!(ŷ,model(g))
        println("Expected: ",y[i],", Obtained: ",argmax(ŷ)-1)
    end
end

#======================================#
#                MAIN                  # 
#======================================#
model = train()
predict(model)
