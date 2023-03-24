using Statistics
using PyPlot
using Flux

#================================================================================#
# INVERTED CART POLE DYNAMICS

function dx(r,F)
    # Parameters
    x = r[1]
    ẋ = r[2]
    θ = r[3]
    ω = r[4]
    ω² = ω^2
    
    m = 0.1     # kg
    M = 1.0     # kg
    g = 9.8067  # m/s²
    l = 0.5     # m    
    
    βx = 0.001   # Linear friction coefficient
    βθ = 0.001   # Angular friction coefficient
    
    d = l*(M + m*sin(θ)^2)
    f1 = (F + m*sin(θ)*(l*ω² - g*cos(θ)) + βθ*m*cos(θ)*ω - βx*ẋ)*l/d
    f2 = (-cos(θ)*(F + m*l*ω²*sin(θ)) + g*(M + m)*sin(θ) - βθ*(m + M)*ω + βx*cos(θ)*ẋ)/d
    
    return [ẋ;f1;ω;f2]
end

# Next state given a vector x = [position, speed, angle, angular speed]
# angle = 0 is perpendicular
function nextState(x,F)
    Δ = 1E-2        # s
    
    K1 = dx(x,F)
    K2 = dx(x .+ 0.5*Δ*K1,F)
    K3 = dx(x .+ 0.5*Δ*K2,F)
    K4 = dx(x .+ Δ*K3,F)
    
    return x + Δ*(K1 .+ 2*K2 .+ 2*K3 + K4)/6
end

# Structure for the environment
mutable struct ICPS
    state       :: Vector{Float64}
    force_max   :: Real
    done        :: Bool
    time        :: Int
    reward      :: Real
end

# Create the environment for the inverted cart pole
function InvertedCartPole()
    MAXIMUM_FORCE = 100.0
    
    return ICPS([0.0,0.0,2π*(rand()-0.5)/25,0],MAXIMUM_FORCE,false,0,0.0)
end

# Reset the environment to a random initial condition
function reset!(env::ICPS)
    setfield!(env,:state, [0,0,2π*(rand()-0.5)/25,0])
    setfield!(env,:done, false)
    setfield!(env,:time, 0)
    setfield!(env,:reward, 0.0)
end

# Simulate
function simulate(env::ICPS,action)
    force = env.force_max*action
    
    # Apply push and let system evolve
    s = nextState(env.state,force)
    
    setfield!(env,:state, s)
    
    return env.state
end    

# Advance the inverted cart pole in time
function step!(env::ICPS,action)
    MAXIMUM_STEPS = 400
    
    if (env.done)
        return env.state,env.reward,env.done
    end
    
    force = env.force_max*action
    
    # Apply push and let system evolve
    s = nextState(env.state,force)
    
    setfield!(env,:state, s)
    setfield!(env,:done, abs(env.state[1]) > 2 || abs(env.state[3]) > 12*π/180 || env.time ≥ MAXIMUM_STEPS)
    
    if (env.done)
        if (env.time ≥ MAXIMUM_STEPS)
            setfield!(env,:reward, 0.0)
        else
            setfield!(env,:reward, -1.0)
        end
    else
        setfield!(env,:reward, cos(env.state[3]) - 0.1*abs(env.state[1]))
    end
    
    setfield!(env,:time,env.time + 1)
    
    return env.state,env.reward,env.done    
end
    

#================================================================================#
# REPLAY BUFFER

mutable struct ReplayBuffer
    state   :: Matrix{Real}
    action  :: Vector{Real}
    reward  :: Vector{Real}
    nstate  :: Matrix{Real}
    done    :: Vector{Bool}
    size    :: Int
    count   :: Int
    idx     :: Int
end

function createReplayBuffer(size)
    return ReplayBuffer(Matrix{Real}(undef,4,size), # state
                        Vector{Real}(undef,size),   # action
                        Vector{Real}(undef,size),   # reward
                        Matrix{Real}(undef,4,size), # next state
                        Vector{Bool}(undef,size),   # done
                        size,                       # size
                        0,                          # count
                        0)                          # index
end

function RBPush!(buffer::ReplayBuffer,state::Vector,action::Real,reward::Real,nstate::Vector,done::Bool)
    buffer.idx = mod1(buffer.idx+1,buffer.size)
    buffer.count = min(buffer.count+1,buffer.size)
    buffer.state[:,buffer.idx] = state
    buffer.action[buffer.idx] = action
    buffer.reward[buffer.idx] = reward
    buffer.nstate[:,buffer.idx] .= nstate
    buffer.done[buffer.idx] = done
end

function RBSample(buffer::ReplayBuffer,batch_size::Int)
    idxs = rand(1:buffer.count,batch_size)
    states = buffer.state[:,idxs]
    actions = buffer.action[idxs]
    rewards = buffer.reward[idxs]
    nstates = buffer.nstate[:,idxs]
    done    = buffer.done[idxs]
    
    return (states,actions,rewards,nstates,done)
end

#================================================================================#
# NEURAL NETWORKS
#
# Initializations: Flux.glorot_normal = Xavier
#                  Flux.kaiming_uniform = He

# Actor network, takes state and estimates the action
function SetActorNetwork()
    m = Chain(
        Dense(4,32,tanh; init = Flux.glorot_normal()),
        Dense(32,64,tanh; init = Flux.glorot_normal()),
        Dense(64,32,tanh; init = Flux.glorot_normal()),
        Dense(32,1,tanh; init = Flux.glorot_normal())
        )
    
    return m
end

# Critic network, takes the state and the action and estimates the Q-value
function SetCriticNetwork()
    m = Chain(
        Dense(5,32,tanh; init = Flux.glorot_normal()),
        Dense(32,64,tanh; init = Flux.glorot_normal()),
        Dense(64,32,tanh; init = Flux.glorot_normal()),
        Dense(32,1; init = Flux.glorot_normal())
        )
    
    return m
end

# Neural Network Approximator
mutable struct NNApprox
    model
    optimiser
end

#================================================================================#
# TRAINING

mutable struct DDPGPolicy 
    behavior_actor  :: NNApprox
    behavior_critic :: NNApprox
    target_actor    :: NNApprox
    target_critic   :: NNApprox
    γ               :: Float64
    τ               :: Float64
end

function buildDDPG()
    ηa = 1.0E-3
    ηc = 2.5E-3
    
    γ = 0.99
    τ = 0.005
    
    behavior_actor = SetActorNetwork()
    behavior_critic = SetCriticNetwork()    
    
    behavior_actor_optimiser = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(ηa))
    behavior_critic_optimiser = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(ηc))
    
    behaviorActorApprox = NNApprox(behavior_actor,behavior_actor_optimiser)
    behaviorCriticApprox = NNApprox(behavior_critic,behavior_critic_optimiser)
    
    target_actor = SetActorNetwork()
    target_critic = SetCriticNetwork()
    
    target_actor_optimiser = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(ηa))
    target_critic_optimiser = Flux.Optimiser(ClipValue(1E-1),Flux.Optimise.ADAM(ηc))
    
    targetActorApprox = NNApprox(target_actor,target_actor_optimiser)
    targetCriticApprox = NNApprox(target_critic,target_critic_optimiser)
    
    return(DDPGPolicy(behaviorActorApprox,behaviorCriticApprox,targetActorApprox,targetCriticApprox,γ,τ))
end 

function DDPGUpdate!(p :: DDPGPolicy, batch)
    # Extract data from batch
    s,a,r,s′,done = batch
    
    a = reshape(a,1,length(a))
    r = reshape(r,1,length(r))
    done = reshape(done,1,length(done))
       
    # Extract data from approximator
    A = p.behavior_actor.model
    C = p.behavior_critic.model
    Aₜ = p.target_actor.model
    Cₜ = p.target_critic.model
    
    Aopt = p.behavior_actor.optimiser
    Copt = p.behavior_critic.optimiser
    
    γ = p.γ
    τ = p.τ
    
    # Compute target Q-value
    a′ = Aₜ(s′)
    x = vcat(s′,a′)
    qₜ = Cₜ(x)
    y = r .+ γ.*(1 .- done).*qₜ
    
    # Compute Critic Gradient
    ∇C = Flux.gradient(params(C)) do
        x = vcat(s,a)
        q = C(x)
        loss = Flux.Losses.mse(y,q)
        loss
    end
    
    # Compute Actor Gradient
    ∇A = Flux.gradient(params(A)) do
        x = vcat(s,A(s))
        loss = -mean(C(x))
        loss
    end
    
    # Update Behavior Networks
    Flux.Optimise.update!(Copt,params(C),∇C)
    Flux.Optimise.update!(Aopt,params(A),∇A)
    
    # Polyak Averaging
    Flux.loadparams!(Aₜ,τ.*Flux.params(A) .+ (1-τ).*Flux.params(Aₜ))
    Flux.loadparams!(Cₜ,τ.*Flux.params(C) .+ (1-τ).*Flux.params(Cₜ))
end
    

# Train a neural network
function TrainDDPG()
    NUM_EPISODES = 1000
    BUFFER_CAPACITY = 60000
    BATCH_SIZE = 128
    
    Policy = buildDDPG()
    R = createReplayBuffer(BUFFER_CAPACITY)
    env = InvertedCartPole()
    
    actor = Policy.behavior_actor.model
    rewardBuffer = [] 
     
    for episode in 1:NUM_EPISODES
        reset!(env)
        totalEpisodeReward = 0
        done = false
            
        while(!done)
            s = env.state
            a = actor(s)[1] + (rand()-0.5)/1000     # Action and noise
                
            # Get reward and next state
            s′,r,done = step!(env,a)
            totalEpisodeReward += r
                      
            # Store transition and sample minibatch
            RBPush!(R,s,a,r,s′,done)
            
            if (R.count ≥ BATCH_SIZE)
                batch = RBSample(R,BATCH_SIZE)
                DDPGUpdate!(Policy,batch)
            end
        end

        # Collect performance
        println("Episode: ",episode,", reward: ", totalEpisodeReward, ", steps: ",env.time)
        push!(rewardBuffer,totalEpisodeReward)
    end
    
    return Policy,rewardBuffer
end

function CheckPolicy(policy :: DDPGPolicy)
    x = []
    y = []
    
    actor = policy.target_actor.model    
    env = InvertedCartPole()
    
    for t in 1:2000
        # Sample {s,a} from actor(a|s)
        s = env.state
        a = actor(s)[1]
        println(a)
        
        ns = simulate(env,a)
        
        # Collect state
        Base.push!(x,ns[1])
        Base.push!(y,ns[3]*180/π)
    end
    
    plot(y)
    xlabel("time [s]")
    ylabel("Angle [degrees]")
    
    ax1 = gca()
    ax2 = ax1.twinx()
    ax2.plot(x,color="orange")
    
    show()
end

#====================================================#
#                       MAIN                         #
#====================================================#

Fmax = 1
policy,R = TrainDDPG()
CheckPolicy(policy)
plot(R)
xlabel("Episode")
ylabel("Rewards")
show()


