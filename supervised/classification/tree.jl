T = ["low"  "low"  "no";
     "low"  "high" "yes";
     "high" "low"  "no";
     "high" "high" "no";
     "fair" "high" "yes";
     "fair" "fair" "yes"]

# Entropy of classes
function Entropy(v)
     s = length(v)
     n = 0
     l = []

     # Find number of classes present
     for vi in v
          if ~(vi in l)
               push!(l,vi)
               n = n + 1
          end
     end

     # Find frequencies
     p = zeros(n)

     for i in 1:s
          for j in 1:n
               if (v[i] == l[j])
                    p[j] = p[j] + 1
               end
          end
     end

     # Compute entropy
     p = p/s
     H = -sum(p.*log2.(p))

     return H
end

# Information gain
function igain(T,p)
     # Find number of elements in attribute
     l = []    # list of attributes
     n = 0     # number of attributes
     for vi in T[:,p]
          if ~(vi in l)
               push!(l,vi)
               n = n + 1
          end
     end

     # Find entropies
     E = 0
     for j in 1:n
          t = []
          f = 0
          for vi in eachrow(T)
               if (vi[p] == l[j])
                    push!(t,vi[3])
                    f = f + 1
               end
          end
          S = Entropy(t)
          f = f / length(T[:,1])
          E = E + S*f
          println(f,"  ",S)
     end

     return Entropy(T[:,3])-E
end

#println(Entropy(T[:,3]))
println(igain(T,1))
println(igain(T,2))
