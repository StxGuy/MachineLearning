using LinearAlgebra
using PyPlot

# Structure for an individual
# and its initialization
mutable struct individual
    chromossome :: Vector{Float64}
end

function individual()
    x = [randn(),randn(),randn()]
    return individual(x)
end

# Computes the fitness of each member of the population.
# In this case it is the norm of the difference between
# each member's chromossome and the target
function Fitness(pop,target)
    N = length(pop)
    r = []
    av = 0
    for p in pop
        F = -norm(p.chromossome - target)
        av += F
        push!(r,F)
    end
    
    return r,av/N
end

# Elitist selection of N elements
function Selection(pop,F,N)
    id = sortperm(F,rev=true)
    new_pop = []
    
    for i in id[1:N]
        push!(new_pop,pop[i])
    end
    
    return new_pop
end

# K-way tournment selecting n individuals
function kway(Fitness,k,n)
    N = length(Fitness)
    
    selection = []
    for i in 1:n
        idx = rand(1:N,k)
        p = argmax(Fitness[idx])
        push!(selection,idx[p])
    end
    
    return selection
end

# Crossover function
function crossover(P1,P2)
    N = length(P1.chromossome)
    C = individual()
    
    for n in 1:N
        if (rand() < 0.5)
            x = P1.chromossome[n]
        else
            x = P2.chromossome[n]
        end
        C.chromossome[n] = x
    end
    
    return C
end

# Creates nchildren individuals using crossover
# between pairs of individuals.
function Reproduction(pop,nchildren)
    F,t = Fitness(pop,target)
    offspring = []
    for i in 1:nchildren
        p1,p2 = kway(F,20,2)
        c = crossover(pop[p1],pop[p2])
        push!(offspring,c)
    end
    
    return vcat(pop,offspring)
end     
    
# Applies mutation to population
function Mutation(pop,η)
    N = length(pop)
    L = length(pop[1].chromossome)
    #α = 1.0
    
    for p in 1:N
        for i in 1:L
            if (rand() < η)
                x = pop[p].chromossome[i]
                y = x + randn()
                pop[p].chromossome[i] = y
            end
        end
    end
    
    return pop
end

# Training function going over selection, reproduction and mutation
function train(pop,target)
    K = length(pop)
    R = 50
    Fi = []
    for it in 1:200
        F,a = Fitness(pop,target)
        pop = Selection(pop,F,(K-R))
        pop = Reproduction(pop,R)
        pop = Mutation(pop,0.005)
        
        push!(Fi,a)
    end
    
    return pop,Fi
end

# Shows the average chromossome value of the population
function showresult(pop)
    r = zeros(3)
    
    for p in pop
        r += p.chromossome
    end
    
    println(r/length(pop))
end
    
#=============== MAIN ==================#
NPop = 100
population = [individual() for i in 1:NPop]
target = [3.1423,2.3567,4.5442]
population,y = train(population,target)
showresult(population)

plot(y)
xlabel("Generation")
ylabel("Average Fitness Value")
show()

    
