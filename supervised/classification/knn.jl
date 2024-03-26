using PyPlot


#=========================#
# PRIORITY QUEUE
#=========================#
struct QueueElement
    index :: Int
    value :: Float64
end

struct PriorityQueue
    max_pos :: Int
    data    :: Array{QueueElement,1}
end

function enqueue!(q::PriorityQueue,priority::Float64,index::Int)
    if length(q.data) < q.max_pos
        push!(q.data,QueueElement(index,priority))
    else
        # find largest element
        max_value = -1E9
        max_pos = 0
        for (idx,el) in enumerate(q.data)
            if el.value > max_value
                max_value = el.value
                max_pos = idx
            end
        end

        # check if it replaces
        if priority < max_value
            q.data[max_pos] = QueueElement(index,priority)
        end
    end
end

#=========================#
# CREATE DATA
#=========================#
μ_redness_orange = 0.5
μ_redness_apple = 0.9
μ_diameter_orange = 8.0
μ_diameter_apple = 6.0

N = 100
training_set = []
xa = []
ya = []
xo = []
yo = []

σ = 0.6

for i in 1:N
    if rand() < 0.65
        μr = μ_redness_orange + randn()*σ
        μd = μ_diameter_orange + randn()*σ
        label = "orange"
        push!(xo,μr)
        push!(yo,μd)
    else
        μr = μ_redness_apple + randn()*σ
        μd = μ_diameter_apple + randn()*σ
        label = "apple"
        push!(xa,μr)
        push!(ya,μd)
    end

    data = [μr,μd,label]
    push!(training_set,data)
end

#=========================#
# KNN
#=========================#
K = 3
x = rand()
y = 4 + rand()*6

# Calculate distance to all elements and keep K nearest neighbors
q = PriorityQueue(K,[])
for i in 1:N
    d = sqrt((x - training_set[i][1])^2 + (y - training_set[i][2])^2)
    enqueue!(q,d,i)
end

# Find the most frequent label among the nearest neighbors
orange = 0
apple = 0
for element in q.data
    if training_set[element.index][3] == "orange"
        global orange += 1
    else
        global apple += 1
    end
end


# Plot and print data
println("Redness: ",x,", Diameter: ",y)
if orange > apple
    println("Orange")
else
    println("Apple")
end

plot(xa,ya,"o",color="red")
plot(xo,yo,"s",color="orange")
plot([x],[y],"*",color="green")
legend(["apple","orange","test"])
xlabel("Redness")
ylabel("Diameter")
show()
