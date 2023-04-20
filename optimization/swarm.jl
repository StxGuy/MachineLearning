using PyPlot

# Agent structure
mutable struct agent
    x       :: Vector{Float64}  # Position
    v       :: Vector{Float64}  # Velocity
    p       :: Vector{Float64}  # Best position
    F       :: Float64          # Fitness
    Fbest   :: Float64          # Best fitness
end

# Swarm structure
mutable struct swarm
    a       :: Vector{agent}    # Agents
    p       :: Vector{Float64}  # Best position
    Fbest   :: Float64          # Best swarm fitness
    xmin    :: Vector{Float64}  # Minimum position
    xmax    :: Vector{Float64}  # Maximum position
    vmin    :: Vector{Float64}  # Minimum velocity
    vmax    :: Vector{Float64}  # Maximum velocity
    c1      :: Float64          # Cognitive parameter
    c2      :: Float64          # Social parameter
    w       :: Float64          # Inertia coefficient
end

# Non-convex test functions: Rosenbrock and Rastrigin
function Rosenbrock(x :: Vector{Float64})
    a = 1.0
    b = 100.0
    
    return (a-x[1])^2 + b*(x[2] - x[1]^2)^2
end

function Rastrigin(x :: Vector{Float64})
    A = 10
    f = 2A
    for i in 1:2
        f += x[i]^2 - A*cos(2π*x[i])
    end
    
    return f
end

# Initialization function for the swarm
function swarm(xi :: Vector{Float64},xa :: Vector{Float64},vi :: Vector{Float64},va :: Vector{Float64},N :: Int)
    Fbest = 1E5
    a = []
    p = xi + rand()*(xa - xi)
    
    for i in 1:N
        r1 = xi[1] + rand()*(xa[1] - xi[1])
        r2 = xi[2] + rand()*(xa[2] - xi[2])
        x = [r1,r2]
        
        r1 = vi[1] + rand()*(va[1] - vi[1])
        r2 = vi[2] + rand()*(va[2] - vi[2])
        v = [r1,r2]
        F = Rosenbrock(x)
        if (F < Fbest)
            Fbest = F
            p = x
        end
        t = agent(x,v,x,F,F)
        
        push!(a,t)
    end
    
    return swarm(a,p,Fbest,xi,xa,vi,va,1.5,0.8,0.8)
end

# Update position and velocity for each agent of the swarm
function updateXV!(s :: swarm)
    for a in s.a
        r1 = rand()
        r2 = rand()
        
        setfield!(a, :v, s.w*a.v + s.c1*r1*(a.p - a.x) + s.c2*r2*(s.p - a.x))
        setfield!(a, :x, a.x + a.v)
    end
end

# Get fitness function. In this case, how close it is from the zero 
# of the function.
function getFitness!(s :: swarm)
    for a in s.a
        setfield!(a, :F, Rosenbrock(a.x))
    end
end

# Find agent closest to the zero of the function and update best
# position for each agent and for the swarm.
function updateBest!(s :: swarm)
    for a in s.a
        if (a.F  < a.Fbest)
            setfield!(a, :Fbest, a.F)
            setfield!(a, :p, a.x)
        end
        if (a.F < s.Fbest)
            setfield!(s, :Fbest, a.F)
            setfield!(s, :p, a.x)
        end
    end
end

# Train swarm.
function train!(s :: swarm)
    # Main loop
    while(s.Fbest > 1E-2)
        updateBest!(s)
        updateXV!(s)
        getFitness!(s)
    end
    
    return s.p
end

# Show results
function showParticles(s :: swarm)
    y_sp = LinRange(-5.12,5.12,100)
    x_sp = LinRange(-5.12,5.12,100)
    map = zeros(100,100)
    for y in 1:100
        for x in 1:100
            map[y,x] = Rosenbrock([x_sp[x],y_sp[y]])
        end
    end
    
    contourf(x_sp,y_sp,map)
    
    for a in s.a
        plot(a.x[1],a.x[2],"o",color="orange")
    end    
    
    show()
end

#=============== MAIN ==============#
s = swarm([-5.12,-5.12],[5.12,5.12],[-1E-1,-1E-1],[1.0,1.0],25)
sol = train!(s)
showParticles(s)
println(s.p,s.Fbest)
