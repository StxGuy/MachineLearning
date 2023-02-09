using Flux
using PyPlot
using Statistics
using LinearAlgebra


# Actor estimates action given a state
function SetActorNetwork()
    m = Chain(
        Dense(4,32,σ),
        Dense(32,5),
        softmax,
    )
        
    return m
end

# Critic estimates value function given a state
function SetCriticNetwork()
    m = Chain(
        Dense(4,8,σ),
        Dense(8,1)
    )
    
    return m
end

# Roulette selection from fitness function
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

# Convert a scalar state into a vector
function embedding(s)
    if (s == 1)
        state = reshape([1,0,0,0],4,1)
        elseif (s == 2)
            state = reshape([0,1,0,0],4,1)
        elseif (s == 3)
            state = reshape([0,0,1,0],4,1)
        else (s == 4)
            state = reshape([0,0,0,1],4,1)
    end    
    
    return state
end

# Clip advantage
function clip(A,ε,r)
    x = copy(r)
    if (A > 0)
        if (r ≥ 1+ε)
            x = 1+ε
        end
    else
        if (r < 1-ε)
            x = 1-ε
        end
    end
    
    return x*A
end

# Returns a zero gradient with appropriate dimensions
function zerograd(network)
    p = params(network)
    t1,t2 = size(p[1])
    x = zeros(t2,1)
    
    ∇ = Flux.gradient(p) do
        0*Flux.mse(critic(x),0)
    end
    
    return ∇
end

# Train a neural network
function TrainNeuralNetwork!(actor,critic,RewardMatrix,StateTransitionMatrix)
    ηa = 1E-3
    ηc = 1E-3
    γ = 0.99
    ε = 0.2
    
    T = 5
    num_actors = 10
    num_interactions = 10000
    K = 10
       
    states = zeros(num_actors,T)
    rewards = zeros(num_actors,T)
    values = zeros(num_actors,T)
    advantages = zeros(num_actors,T)
    actions = zeros(num_actors,T)
    probs = zeros(num_actors,T)
    
    actor_optimizer = Flux.Optimise.ADAM(ηa)
    critic_optimizer = Flux.Optimise.ADAM(ηc)
    
    loss = []
    
    for interaction in 1:num_interactions
        avR = 0
        avS = 0
        avA = 0
        for act in 1:num_actors             # Run the policy for many actors in parallel
            # Run policy for T timesteps
            s = rand([1,2,4])               # Initial state
            R = 0                           # Initial reward
            V = critic(embedding(s))[1,1]   # Initial value
            
            steps = 0                       # Steps counter
            for t in 1:T
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
                                
                # Store values
                states[act,t] = s
                rewards[act,t] = r
                values[act,t] = V
                advantages[act,t] = A
                actions[act,t] = action
                probs[act,t] = actor_output⋅Flux.onehot(action,1:5)
                
                # Move to next state
                s = ns
            end
        end
        
        M = num_actors*T
        
        # Updates
        for epoch in 1:K
            ∇c = zerograd(critic)
            ∇a = zerograd(actor)

            for minibatch in 1:M
                act = rand(1:num_actors)
                t = rand(1:T)
                
                s = states[act,t]
                r = rewards[act,t]
                V = values[act,t]
                A = advantages[act,t]
                action = actions[act,t]
                πθo = probs[act,t]
                
                # critic gradient
                ∇c .+= Flux.gradient(params(critic)) do
                    Flux.mse(critic(embedding(s)),r+γ*V)
                end
                                                                                                
                # actor gradient
                ∇a .+= Flux.gradient(params(actor)) do
                    πθ = actor(embedding(s))⋅Flux.onehot(action,1:5)
                    r = exp(log(πθ) - log(πθo))
                    -clip(A,ε,r)
                end
            end
              
            Flux.Optimise.update!(critic_optimizer,params(critic),∇c./M)  
            Flux.Optimise.update!(actor_optimizer,params(actor),∇a./M)
        end
        
        avR = mean(rewards)
        avS = mean(states)
        avA = mean(advantages)
        
        push!(loss,avR)
        println("Epoch ",interaction,", average reward: ",avR,", average steps: ",avS, ", <A>: ",avA)
    end        
    
    return loss
end

function CheckPolicy(actor)
    for (s,t) in zip(1:4,["A","B","C","D"])
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
        if (x == 5)
            a = "stay"
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
# up    = [1 0 0 0 0]
# down  = [0 1 0 0 0]
# left  = [0 0 1 0 0]
# right = [0 0 0 1 0]
# stay  = [0 0 0 0 1]

actor = SetActorNetwork()
critic = SetCriticNetwork()

# Reward matrix
Rm = [-1 -1  -1  0  0;
      -1  0   0 -1  0;
      -1 -1  -1  0 10;
       0 -1 100 -1  0]

# Next state matrix      
M = [1 1 1 2 1;
     2 4 1 2 2;
     3 3 3 4 3;
     2 4 3 4 4]

R = TrainNeuralNetwork!(actor,critic,Rm,M)
CheckPolicy(actor)
plot(R)
xlabel("Episode")
ylabel("Rewards")
show()


