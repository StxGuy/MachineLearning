using LinearAlgebra
using Statistics
using Images
using PyPlot

function createdata(T)
    img = load("img2.png")
    y = zeros(100,100)

    for j in 1:100
        for i in 1:100
            r = red(img[i,j])
            g = green(img[i,j])
            b = blue(img[i,j])
            
            y[i,j] = 0.2126*r + 0.7152*g + 0.0722*b
        end
    end

    noise = rand(100,100)
    
    y1 = reshape(y,(10000,1))
    y2 = reshape(noise,(10000,1))
        
    S = hcat(y1,y2)'
    
    A = [0.70 0.30; 0.40 0.60]
    R = A*S
    
    return R,S
end

function sphere(R)
    #---- Sphering ----
    # Centering
    R = (R .- mean(R,dims=2))

    # ZCA Whitening
    C = R*R'/Te
    v = eigvals(C)
    U = eigvecs(C)

    Σ = Diagonal(1 ./ sqrt.(v))

    T = U*Σ*U'
    Rw = T*R

    return Rw
end

g(u) = 4u^3
h(u) = 12u^2

function estimate(Rw)
    η = 0.01
    
    #---- ICA ----
    V = rand(2,2)
    v1 = V[1,:]
    v2 = V[2,:]
    
    for it in 1:100
        t = v1'*Rw
        λ = (g.(t)*t')[1,1]
        f = Rw*g.(t')-λ*v1
        J = Rw*diagm(h.(t)[1,:])*Rw'-λ*I
        p = v1 - η*inv(J)*f
        v1 = p/norm(p)
        
        t = v2'*Rw
        λ = (g.(t)*t')[1,1]
        f = Rw*g.(t')-λ*v2
        J = Rw*diagm(h.(t)[1,:])*Rw'-λ*I
        p = v2 - η*inv(J)*f
        v2 = p/norm(p)
        v2 = v2 - (v2'*v1)*v1
        v2 = v2/norm(v2)
    end

    V = hcat(v1,v2)
    println(V)
    Sh = V'*Rw
    
    return Sh
end

#=========================================#
#                 MAIN                    #
#=========================================#
Te = 150

R,S = createdata(Te)
Rw = sphere(R)
Sh = estimate(Rw)


#--- Plotting ---
imshow(reshape(R[1,:],(100,100)),cmap="gray")
show()
imshow(reshape(R[2,:],(100,100)),cmap="gray")
show()
imshow(reshape(S[1,:],(100,100)),cmap="gray")
show()
imshow(reshape(S[2,:],(100,100)),cmap="gray")
show()
imshow(reshape(Sh[1,:],(100,100)),cmap="gray")
show()


