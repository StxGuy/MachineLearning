using PyPlot
using Statistics
using LinearAlgebra

# Synthetic training data
function createData(N)
    a0 = 1.29
    a1 = 1.43
    a2 = 1.15
    a = [a0,a1,a2]

    x0 = []
    x1 = []
    x2 = []
    for i in 1:N
        push!(x0, 1)
        push!(x1, randn())
        push!(x2, randn())
    end
    X = hcat(x0,x1,x2)
    y = X*a + randn(N,1)

    return X,y
end

# Regression
function LARS(X,y)
    η = 0.05
    a = [0.0,0.0,0.0]
    r = y

    x1 = X[:,2]
    x2 = X[:,3]

    y1 = []
    y2 = []
    y3 = []
    x = []
    lst = []

    r2 = 0
    r1a = []
    r2a = []

    for it in 1:10000
        ρ = [sum(r), x1⋅r, x2⋅r]
        i = argmax(abs.(ρ))

        if (~(i in lst))
            push!(lst,i)
        end

        # Find equiangular direction
        d = 0
        for j in lst
            d = d + η*sign(ρ[j])
        end
        d = d / length(lst)

        # Update all coefficients in the list and residuals
        for j in lst
            a[j] = a[j] + d
            r = r - d*X[:,j]
        end

        push!(x,norm(a))
        push!(y1,a[1])
        push!(y2,a[2])
        push!(y3,a[3])

        yh = X*a
        r2 = cor(yh,r)[1]
        push!(r2a,r2)
        push!(r1a,1-(r⋅r)/(y'*y)[1,1])
    end

    println(a)
    println(r2)

    return x,y1,y2,y3,r1a
end

#===============================================#
#                    MAIN                       #
#===============================================#
X,y = createData(100)
x,y1,y2,y3,r1a = LARS(X,y)

subplot(121)
plot(x,y1,color="lightgray")
plot(x,y2,color="gray")
plot(x,y3,color="darkgray")
axis([0,2.5,0,1.6])
xlabel("|a|")
ylabel("a")

subplot(122)
plot(x,r1a,color="gray")
xlabel("Step")
ylabel("R^2")
axis([0,2.5,0,1])

show()
        
        
