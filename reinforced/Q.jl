# ε-greedy algorithm for exploration/exploitation
# returns action
function ε_greedy(s,ε,Q)
    r = rand()
    
    if (r < ε)      # Exploration
        a = rand(1:size(Q,2))
    else            # Exploitation
        a = argmax(Q[s,:])
    end
    
    return a
end

# Q-learning algorithm
function Q_Learning(reward,nstate,α,γ)
    ns,na = size(nstate)
    Q = rand(ns,na)

    for it in 1:500
        s = rand(1:ns)

        while(s ≠ 3)
            a = ε_greedy(s,0.1,Q)
            r = reward[s,a]
            sn = nstate[s,a]
            
            Q[s,a] = (1-α)*Q[s,a] + α*(r+γ*maximum(Q[sn,:]))
            s = sn
        end
    end
    
    return Q
end

# Simple function to display matrix
function showmat(M)
    for m in eachrow(M)
        println(m)
    end
end

#====================================================#
#          Main
#====================================================#
# A = 1, B = 2, C = 3, D = 4
# S = up, down, left, right

# Reward matrix
R = [-1 -1  -1  0;
     -1  0   0 -1;
     -1 -1  -1  0;
      0 -1 100 -1]

# Next state matrix      
M = [1 1 1 2;
     2 4 1 2;
     3 3 3 4;
     2 4 3 4]

Q = Q_Learning(R,M,0.7,0.95)
showmat(Q)
