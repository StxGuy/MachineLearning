using PyPlot

# Edge: [node 1, node2, age]
# Node: [x, y, error, status]
#   status = alive(true)/dead(false)

mutable struct graph
    list_of_edges   :: Vector
    list_of_nodes   :: Vector
end

# Constructor
function beginGraph()
    return graph([],[])
end

# Euclidean distance
function EuclideanDistance(x,y)
    n = length(x)
    s = 0

    for i in 1:n
        s = s + (x[i]-y[i])^2
    end

    return s
end

# Draw network
function drawNet(G::graph)
    for i in 1:length(G.list_of_edges)
        p1 = G.list_of_edges[i][1]
        p2 = G.list_of_edges[i][2]
        if (p1 != 0 && p2 != 0)
            node1 = G.list_of_nodes[p1]
            node2 = G.list_of_nodes[p2]
            if (node1[1] != 0 && node1[2] != 0 && node2[1] != 0 && node2[2] != 0)
                x = [node1[1],node2[1]]
                y = [node1[2],node2[2]]
                plot(x,y,color="gray",alpha=0.65)
            end
        end
    end
end

#==================== NODE FUNCTIONS ===================#

# Find two closest nodes to a vector
function findClosestNodes(G::graph,v::Vector)
    d1 = 1E5
    d2 = 1E5
    s1 = 0
    s2 = 0

    for i in 1:length(G.list_of_nodes)
        if (G.list_of_nodes[i][4] == true)
            d = EuclideanDistance(G.list_of_nodes[i][1:2],v)

            if (d < d1)
                d1 = d
                s1 = i
            end
            if (d < d2 && d > d1)
                d2 = d
                s2 = i
            end
        end
    end

    return s1,s2,d1
end

# Move node towards vector
function moveNode(G::graph,s1::Integer,x::Vector)
    η1 = 0.1
    η2 = 0.01

    # Move main node
    G.list_of_nodes[s1][1:2] = G.list_of_nodes[s1][1:2] + η1*(x-G.list_of_nodes[s1][1:2])

    # Move topological neighbors
    for i in 1:length(G.list_of_edges)
        edge = G.list_of_edges[i]
        if (edge[1] == s1)
            s2 = edge[2]
            G.list_of_nodes[s2][1:2] = G.list_of_nodes[s1][1:2] + η2*(x-G.list_of_nodes[s2][1:2])
        end
        if (edge[2] == s1)
            s2 = edge[1]
            G.list_of_nodes[s2][1:2] = G.list_of_nodes[s2][1:2] + η1*(x-G.list_of_nodes[s2][1:2])
        end
    end
end

# Find number of neighbors
function number_of_neighbors(G::graph,n::Integer)
    nei = 0
    for i in 1:length(G.list_of_edges)
        if (G.list_of_edges[i][1] == n || G.list_of_edges[i][2] == n)
            nei = nei + 1
        end
    end

    return nei
end

# Insert a node
function insertNode(G::graph,v::Vector)
    r = 0

    # Empty list
    if (length(G.list_of_nodes) == 0)
        p = [v[1],v[2],0,true]
        push!(G.list_of_nodes,p)
        r = 1
    else
        i = 1
        # Search for a spot by scanning the list of edges
        L = length(G.list_of_nodes)
        while(G.list_of_nodes[i][4] == true && i < L)
            i = i + 1
        end

        # if i < L, a spot was found
        # if i == L, either L is a spot or it has reached
        # the end of the list
        if (i < L || (i == L && G.list_of_nodes[L][4] == false))
            # Found a dead node
            G.list_of_nodes[i][1:2] = v
            G.list_of_nodes[i][3] = 0
            G.list_of_nodes[i][4] = true
            r = i
        else
            p = [v[1],v[2],0,true]
            push!(G.list_of_nodes,p)
            r = L + 1
        end
    end

    return r
end

#================= EDGE FUNCTIONS ==================#

# Remove edges older than a certain age
function removeOldEdges(G::graph,thresholdAge)
   for i in 1:length(G.list_of_edges)
        if (G.list_of_edges[i][3] > thresholdAge)
            p1 = G.list_of_edges[i][1]
            p2 = G.list_of_edges[i][2]
            G.list_of_edges[i] = [0,0,0]

            # If nodes become orphans, delete them
            if (number_of_neighbors(G,p1) == 0)
                G.list_of_nodes[p1][4] = false
            end
            if (number_of_neighbors(G,p2) == 0)
                G.list_of_nodes[p2][4] = false
            end
        else
        end
    end
end

# Age edges
function ageEdges(G::graph,s1::Integer)
    for i in 1:length(G.list_of_edges)
        edge = G.list_of_edges[i]
        if (edge[1] == s1 ||edge[2] == s1)
            G.list_of_edges[i][3] = G.list_of_edges[i][3] + 1
        end
    end
end

# Remove connection between nodes
function removeEdge(G::graph,a::Integer,b::Integer)
    i = 1
    st = false

    while(!st)
        c1 = G.list_of_edges[i][1] == a || G.list_of_edges[i][2] == b
        c2 = G.list_of_edges[i][2] == a || G.list_of_edges[i][1] == b
        if (c1 || c2)
            G.list_of_edges[i] = [0,0,0]
            st = true
        else
            i = i + 1
            if (i > length(G.list_of_edges))
                st = true   # Not found
            end
        end
    end
end

# Connect two nodes
function connectNodes(G::graph,s1::Integer,s2::Integer)
    L = length(G.list_of_edges)

    # Empty list
    if (L == 0)
        push!(G.list_of_edges,[s1,s2,0])
    # Scan list of edges
    else
        i = 1
        while(G.list_of_edges[i][1] != 0 && i < L)
            i = i + 1
        end

        # if i < L, a spot was found
        # if i == L, a spot may be found or the end of the list was reached
        if (i < L || (i == L && G.list_of_edges[i][1] == 0))
            G.list_of_edges[i] = [s1,s2,0]
        else
            push!(G.list_of_edges,[s1,s2,0])
        end
    end
end


#================= ERROR FUNCTIONS ==================#

# Find node with maximum error and its neighbor with maximum error
function findMaxError(G::graph)
    sm = 0
    sn = 0
    maxErr = -1E5

    # Find node with biggest error
    for i in 1:length(G.list_of_nodes)
        node = G.list_of_nodes[i]
        if (node[4] == true)
            if (node[3] > maxErr)
                maxErr = node[3]
                sm = i
            end
        end
    end

    # Find topological neighbor with biggest error
    maxErr = -1E5
    for i in 1:length(G.list_of_edges)
        if (G.list_of_edges[i][1] == sm)
            s2 = G.list_of_edges[i][2]
            if (G.list_of_nodes[s2][3] > maxErr)
                maxErr = G.list_of_nodes[s2][3]
                sn = s2
            end
        end
        if (G.list_of_edges[i][2] == sm)
            s2 = G.list_of_edges[i][1]
            if (G.list_of_nodes[s2][3] > maxErr)
                maxErr = G.list_of_nodes[s2][3]
                sn = s2
            end
        end
    end

    return sm,sn
end

# Add error to node
function addError(G::graph,s1::Integer,E::Real)
    G.list_of_nodes[s1][3] = E
end

# Set error of a node equal to the error of another
function setError(G::graph,dst::Integer,src::Integer)
    G.list_of_nodes[dst][3] = G.list_of_nodes[src][3]
end

# Discount error of a node
function discountError(G::graph,node::Integer,λ::Real)
    G.list_of_nodes[node][3] = G.list_of_nodes[node][3]*λ
end

# Discount all errors
function discountAllErrors(G::graph,λ::Real)
    for i in 1:length(G.list_of_nodes)
        G.list_of_nodes[i][3] = G.list_of_nodes[i][3]*λ
    end
end

#===================== MAIN ======================#

# Initial guess
G = beginGraph()
for i in 1:50
    insertNode(G,[2*(rand()-0.5),2*(rand()-0.5)])
end

# Main loop
counter = 0

for it in 1:10000
# Draw a vector x from annular submanifold
    r = 2 + rand()
    θ = 2π*rand()
    xi = r*cos(θ)
    yi = r*sin(θ)
    x = [xi,yi]

    # Neuronal gas algorithm
    s1,s2,ε = findClosestNodes(G,x)
    moveNode(G,s1,x)
    connectNodes(G,s1,s2)
    ageEdges(G,s1)
    addError(G,s1,ε)
    removeOldEdges(G,5)

    global counter += 1
    if (counter > 10)
        sm,sn = findMaxError(G)
        v = 0.5*(G.list_of_nodes[sm][1:2] + G.list_of_nodes[sn][1:2])
        sh = insertNode(G,v)
        removeEdge(G,sn,sm)
        connectNodes(G,sn,sh)
        connectNodes(G,sm,sh)
        discountError(G,sn,0.9)
        discountError(G,sm,0.9)
        setError(G,sh,sn)
        counter = 0
    end
    discountAllErrors(G,0.9)
end

drawNet(G)
show()
