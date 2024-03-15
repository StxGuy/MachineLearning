using PyPlot
using Images
using Statistics
using LinearAlgebra

function distance(r1,r2)
    return norm(r1 - r2)
end


function KMeans(filename,Nc)
    img = load(filename)
    img = channelview(img)

    Lc,Ly,Lx = size(img)
    Nc = 16

    # Initialization
    C = zeros(3,16)
    for i in 1:16
        C[:,i] = [1.0/i,1.0/i,1.0/i]
    end

    imgc = zeros(Ly,Lx)
    for steps = 1:10
        # Assignment
        for j in 1:Lx, i in 1:Ly
            d_min = 1E9
            c_min = 0

            for c in 1:Nc
                d = distance(img[:,i,j], C[:,c])

                if d < d_min
                    d_min = d
                    c_min = c
                end
            end

            imgc[i,j] = c_min
        end

        # Recentering
        C = zeros(Lc,Nc)
        Sc = zeros(Nc)
        for j in 1:Lx, i in 1:Ly
            C[:,Int(imgc[i,j])] += img[:,i,j]
            Sc[Int(imgc[i,j])] += 1
        end

        for i in 1:Nc
            if Sc[i] > 0
                C[:,i] = C[:,i]/Sc[i]
            end
        end
    end

    # Reconstruct
    rec = zeros(Ly,Lx,Lc)
    for j in 1:Lx, i in 1:Ly
        rec[i,j,:] = C[:,Int(imgc[i,j])]
    end

    return rec
end

# Load image
rec = KMeans("NAU.jpg",16)
imshow(rec)
show()

