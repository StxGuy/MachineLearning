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

function SetJoined(nstates,nactions)
    m = Chain(x->vcat(x[1],x[2]),
        Dense(nstates+nactions,32,tanh; init = Flux.glorot_normal()),
        Dense(32,64,tanh; init = Flux.glorot_normal()),
        Dense(64,1;init = Flux.glorot_normal())
    )
    
    return m
end

# An approximator for the neural network with an optimiser
mutable struct NNApprox
    model       :: Chain
    target      :: Chain
    optimiser   :: Flux.Optimiser
end

# Returns an approximator
function SetAppr(η)
    model = SetNet(1,2)
    target = SetNet(1,2)
    
    Flux.loadparams!(target,Flux.params(model))
    
    op = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(η))
   
   return NNApprox(model,target,op)
end

function SetApprJ(η)
    model = SetJoined(1,2)
    target = SetJoined(1,2)
    
    Flux.loadparams!(target,Flux.params(model))
    
    op = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(η))
    
    return NNApprox(model,target,op)
end

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

function train!(A::NNApprox,C::NNApprox,R::ReplayBuffer)
    γ = 0.95
    S = sample(R,100)
    
    for (s,a,r,s′,d) in S
        y = r + γ*(1-d)*C.target(([s′],A.target([s′])))[1]
        
        ∇c = Flux.gradient(params(C.model)) do
            loss = (y - C.model(([s],a))[1])^2
        end
                
        ∇a = Flux.gradient(params(A.model)) do
            loss = -C.model(([s],A.model([s])))[1]
        end
        
        Flux.Optimise.update!(C.optimiser,Flux.params(C.model),∇c)
        Flux.Optimise.update!(A.optimiser,Flux.params(A.model),∇a)
    end
end


# Replay buffer
R = ReplayBuffer(40000)

# Approximator for Q-value
Actor = SetAppr(1E-3)
Critic = SetApprJ(2E-3)

γ = 0.95
α = 0.1
ζ = 1       # Exploration rate
τ = 0.005   # Target update rate
g = 9.8067  # [m/s²]
m = 1       # [kg]

y_plot = []
for tc in 1:200000
    # Begin
    s = 5*(rand()-0.5)
    totalReward = 0
    
    ti = 0
    while(abs(s - 3) > 1E-3 && ti < 10)
        ti += 1
        
        a = Actor.model([s])
        
        # Environment
        θ = (a[1]*(1-ζ) + rand()*ζ)*π  
        vo = (a[2]*(1-ζ) + rand()*ζ)*10
        
        t = (2*vo/g)*sin(θ)
        x = vo*t*cos(θ)
        s′ = s + x
        r = -abs(s′-3)
        totalReward += r
                
        d = false
        if (abs(s′-3) ≤ 1E-3 || ti ≥ 10)
            d = true
        end

        store!(R,(s,a,r,s′,d))
        s = s′
                
        # Reduce exploration
        global ζ = max(1E-6,ζ-1E-6)
    end

    if (tc % 100 == 0)
        push!(y_plot,totalReward)
        println("Reward: ",totalReward,", Exploration: ",ζ,", Steps: ",ti, ", Time: ",tc)
    end
    
    train!(Actor,Critic,R)
    Flux.loadparams!(Actor.target,τ.*Flux.params(Actor.model) .+ (1-τ).*Flux.params(Actor.model))
    Flux.loadparams!(Critic.target,τ.*Flux.params(Critic.model) .+ (1-τ).*Flux.params(Critic.model))
end

function showmat(M)
    for m in eachrow(M)
        println(m)
    end
end

#showmat(Q)
for s in 1:4
    a = Actor.target([s])
    θ = a[1]*π
    vo = a[2]*10
    
    t = (2*vo/(m*g))*sin(θ)
    x = s + vo*t*cos(θ)
    
    println("xo: ",s,", θ: ",θ*180/π,", vo: ",vo, ", x: ",x)
end

plot(y_plot)
show()

