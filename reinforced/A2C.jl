using PyPlot
using Flux


# Initialize Neural Network
function SetActorNetwork()
    m = Chain(
        Dense(4,32,σ),
        Dense(32,4),
        softmax,
    )
        
    return m
end

function SetCriticNetwork()
    m = Chain(
        Dense(4,8,σ),
        Dense(8,1)
    )
    
    return m
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
function TrainNeuralNetwork!(actor,critic,RewardMatrix,StateTransitionMatrix)
    ηa = 1E-5
    ηc = 1E-3
    num_episodes = 100
    num_epochs = 1000
    γ = 0.99
    
    actor_optimizer = Flux.Optimise.ADAM(ηa)
    critic_optimizer = Flux.Optimise.ADAM(ηc)
    
    loss = []
    
    for epoch in 1:num_epochs
        avR = 0
        avS = 0
        avA = 0
        for episode in 1:num_episodes
            s = rand([1,2,4])
            R = 0
            steps = 0
            V = critic(embedding(s))[1,1]
            errorV = 0
            
            while(s ≠ 3 && steps < 5)
                # Sample {s,a} from actor(a|s)
                steps += 1
                actor_output = actor(embedding(s))
                action = roulette_selection(actor_output)
                
                # Get reward and next state
                r = RewardMatrix[s,action]
                R += γ*r
                ns = StateTransitionMatrix[s,action]
                
                # Calculate advantage
                lastV = V
                V = critic(embedding(ns))[1,1]
                A = r + γ*V - lastV     # TD Error
                avA += A

                # Update critic
                ∇c = Flux.gradient(params(critic)) do
                    Flux.mse(critic(embedding(s)),r+γ*V)
                end
                Flux.Optimise.update!(critic_optimizer,params(critic),∇c)                
                
                # Update actor
                ∇a = Flux.gradient(params(actor)) do
                    -sum(log.(actor(embedding(s))).*Flux.onehot(action,1:4))*A
                end
                Flux.Optimise.update!(actor_optimizer,params(actor),∇a)    
                                                
                # Update state
                s = ns
            end
       
            avR += R
            avS += steps            
        end         
               
        
        avR /= num_episodes
        avS /= num_episodes
        avA /= num_episodes
        
        push!(loss,avS)
        println("Epoch ",epoch,", average reward: ",avR,", average steps: ",avS, ", <A>: ",avA)
    end        
    
    return loss
end

function CheckPolicy(actor)
    for (s,t) in zip([1,2,4],["A","B","D"])
        actor_output = actor(embedding(s))
        x = argmax(actor_output[:,1])
                
        if (x == 1)
            a = "up"
        end
        if (x == 2)
            a = "down"
        end
        if (x == 3)
            a = "left"
        end
        if (x == 4)
            a = "right"
        end
        
        println(t,": ",a)
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

actor = SetActorNetwork()
critic = SetCriticNetwork()

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

R = TrainNeuralNetwork!(actor,critic,Rm,M)
CheckPolicy(actor)
plot(R)
xlabel("Episode")
ylabel("Steps")
show()


