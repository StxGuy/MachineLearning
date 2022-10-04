using PyPlot

# Stump structure
mutable struct stumpStructure
     attribute  :: Int
     out        :: Dict{String,Int}
end

# Fit a binary stump
function fitBinaryStump!(T, s :: stumpStructure)
    # Find labels for attribute a
    feature = []
    for t in eachrow(T)
        if ~(t[s.attribute] in feature)
            push!(feature,t[s.attribute])
        end
    end
    N = length(feature)
    
    # Find scores
    score = zeros(N)
    for t in eachrow(T)
        for n in 1:N
            if (t[s.attribute] == feature[n])
                score[n] += t[4]
            end
        end
    end
    
    # Fit 
    for n in 1:length(feature)
        if (score[n] > 0)
            s.out[feature[n]] = 1
        else
            s.out[feature[n]] = -1
        end
    end
end

# Gives the output of a stump
function stump(s :: stumpStructure,c1,c2,c3)
    if (s.attribute == 1)
        r = s.out[c1]
    elseif(s.attribute == 2)
        r = s.out[c2]
    else
        r = s.out[c3]
    end
    
     return r
end

# Binary Classifier for the Forest
function Classifier(c1,c2,c3,Ensemble,alphas)
     C = 0
     for (s,α) in zip(Ensemble,alphas)
          C = C + stump(s,c1,c2,c3)*α
     end

     return sign(C)
end

# Add another stump to the forest
function Add(Ensemble,alphas,weights,p,T)
     s = stumpStructure(p,Dict())
     fitBinaryStump!(T,s)

     # Find w_{i,n}
     misclassified = 0
     total = 0
     ξ_list = []
     for (xi,w) in zip(eachrow(T),weights)
          k = stump(s,xi[1],xi[2],xi[3])

          # Find misclassification
          y = xi[4]
          if (k ≠ y)
               misclassified += w
               push!(ξ_list,1)
          else
               push!(ξ_list,-1)
          end
          total += w
     end
     # Error rate
     ε = misclassified/total

     α = 0.5*log((1-ε)/ε)

     # Update weights
     nw = []
     Z = 0
     for (w,ξ) in zip(weights,ξ_list)
         el = w*exp(ξ*α)
         Z = Z + el
         push!(nw,w*exp(ξ*α))
     end
     nw = nw/Z
    
     return vcat(Ensemble,s), vcat(alphas,α), nw
end

# Make predictions
function Forest(ens,als,T)
    for t in eachrow(T)
        c = Classifier(t[1],t[2],t[3],ens,als)
        println(c, " should be ",t[4])
    end
end

# Calculate performance
function performance(ens,als,T)
    right = 0
    for t in eachrow(T)
        c = Classifier(t[1],t[2],t[3],ens,als)
        
        if (c == t[4])
            right += 1
        end
    end
    
    return right/length(T[:,1])
end

#------ MAIN FUNCTION -----#
function main()
    ensemble = []
    alphas = []
    weights = [0.2 0.2 0.2 0.2 0.2]
    
    T = ["a" "c" "e" 1;
          "b" "d" "e" 1;
          "a" "c" "e" -1;
          "b" "c" "e" -1;
          "a" "d" "g" 1]
    
    y = []
    for it in [1 3 2]
        ensemble,alphas,weights = Add(ensemble,alphas,weights,it,T)
        push!(y,performance(ensemble,alphas,T))
    end
    
    Forest(ensemble,alphas,T)
    println(alphas)
    
    plot(y,"*-")
    show()
end

main()


