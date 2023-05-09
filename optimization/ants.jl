#---------------------
# Ant-agent structure
#---------------------
mutable struct antAgent
    node    :: Int
    memory  :: Vector
end

# Ant-agent Constructor
function antAgent(node :: Int)
    return antAgent(node,[])
end

#------------------
# Colony structure
#------------------
mutable struct antColony
    number_of_ants  :: Int          # Number of ants
    target_node     :: Int          # Target node
    τ               :: Matrix       # Pheromone trail level
    A               :: Matrix       # Adjacency matrix
    ants            :: Vector       # Ants
end    

# Colony constructor
function build_colony(A :: Matrix, target_node :: Int, number_of_ants :: Int)
    N, = size(A)
    τ = zeros(N,N)
    
    # Place ants randomly in graph
    ants = []
    for i in 1:number_of_ants
        push!(ants,antAgent(rand(1:N)))
    end
    
    return antColony(number_of_ants,target_node,τ,A,ants)
end

# Standard roulette selection
function roulette_selection(fitness)
    if (sum(fitness) == 0)
        return rand(1:length(fitness))
    else
        total_fitness = sum(fitness)
        r = rand()*total_fitness
        
        i = 0
        while(r ≥ 0)
            i += 1
            r -= fitness[i]
        end
        
        return i
    end
end

# Find neighbors given a node
function find_neighbors(A,i)
    N, = size(A)
    
    k = []
    for j in 1:N
        if (A[i,j] ≠ 0)
            push!(k,j)
        end
    end
    
    return k
end

# Construct solutions phase
function construct_solutions(colony :: antColony)
    α = 1.5
    β = 1.5
    τo = 1.0
    po = 0.1
    ρ = 0.05
    
    N, = size(colony.A)
    
    # Erase memory and start on a random location
    for ant in colony.ants
        setfield!(ant,:memory,[])
        setfield!(ant,:node,rand(1:N))
    end
    
    # Explore solution space until all ants find target node
    complete_tours = 0
    while(complete_tours < colony.number_of_ants)
        complete_tours = 0
        for ant in colony.ants
            neighbors = find_neighbors(colony.A,ant.node)
            neighbors = setdiff(neighbors,ant.memory)   # Neighbors - memory
            
            if (isempty(neighbors) == false && ant.node ≠ colony.target_node)
                fitness = []
                for j in neighbors
                    η = 1/colony.A[ant.node,j]          # Attractiveness
                    p = (colony.τ[ant.node,j]^α)*(η^β)  
                    push!(fitness,p)
                end
                
                # Move to new node stochastically
                if (rand() ≤ po)
                    k = neighbors[argmax(fitness)]
                else
                    k = neighbors[roulette_selection(fitness)]
                end
                setfield!(ant,:node,k)
                
                colony.τ[ant.node,k] = (1.0-ρ)*colony.τ[ant.node,k] + ρ*τo
                setfield!(colony,:τ,colony.τ)
                
                # Add node to memory
                memory = ant.memory ∪ k
                setfield!(ant,:memory,memory)
            else
                complete_tours += 1
            end
        end
    end
end

# Update pheromone levels
function update_pheromone(colony :: antColony)
    ρ = 0.01
    Q = 10.0
        
    N, = size(colony.A)
    for ant in colony.ants
        for i in 1:(N-1)
            p_from = findfirst(ant.memory .== i)
            for j in (i+1):N
                p_to = findfirst(ant.memory .== j)
                if (isnothing(p_from) == false && isnothing(p_to) == false)
                    if (p_to - p_from == 1)
                        Δ = Q/length(ant.memory)
                    else
                        Δ = 0.0
                    end                
                else
                    Δ = 0.0
                end
                    
                colony.τ[i,j] = (1.0-ρ)*colony.τ[i,j] + Δ
                setfield!(colony,:τ,colony.τ)
            end
        end
    end
end
        
#====================== MAIN =======================#        
A = [0 1 1 0 0;
     1 0 1 0 0;
     1 1 0 1 1;
     0 0 1 0 1;
     0 0 1 1 0]
colony = build_colony(A,5,100)

# Main loop
for it in 1:10
    construct_solutions(colony)
    update_pheromone(colony)
end

# Print results
for i in 1:5
    println(colony.τ[i,:])
end
