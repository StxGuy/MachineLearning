
program kmeans
    implicit none
    
    integer,parameter :: width=410,height=512
    
    real                :: C(64,3)
    integer             :: cluster_size(64)
    real                :: d_min,d
    integer             :: i_min
    integer             :: i,j,k,t,ac,loop
    integer             :: label(height,width)
    integer             :: fig_in(height,width,3)
    real                :: fig_out(height,width,3)
    character(len=50)   :: fname
    logical             :: existent
    
    
    ! Load image
    open(1,file="fig_in.dat",status="old")
    
    do j = 1,width
    do i = 1,height
        read(1,*) fig_in(i,j,:)
    end do
    end do
    
    close(1)    
    
    ! Set initial medoids
    ac = 1
    do k = 1,4
        do j = 1,4
            do i = 1,4
                C(ac,3) = k*64
                C(ac,2) = j*64
                C(ac,1) = i*64
                ac = ac + 1
            end do
        end do
    end do

    do loop = 1,10
        ! Clusterize data
        do j = 1,width
        do i = 1,height
            d_min = 1E5
            do k = 1,64
                ! L2
                d = 0
                do t = 1,3
                    d = d + (fig_in(i,j,t) - C(k,t))**2
                end do
                ! Find nearest cluster
                if (d < d_min) then
                    d_min = d
                    i_min = k
                end if
            end do
            label(i,j) = i_min
        end do
        end do
            
        ! Recalculate centroids
        C = 0.0
        cluster_size = 0
        do j = 1,width
        do i = 1,height
            C(label(i,j),:) = C(label(i,j),:) + fig_in(i,j,:)
            cluster_size(label(i,j)) = cluster_size(label(i,j)) + 1
        end do
        end do
        
        do i = 1,64
            C(i,:) = C(i,:)/cluster_size(i)
        end do
    end do
    
    ! Reconstruct image
    do j = 1,width
    do i = 1,height
        fig_out(i,j,:) = C(label(i,j),:)
    end do
    end do

    ! Save image
    fname = "fig_out.dat"
    inquire(file=fname,exist=existent)
    
    if (existent) then
        open(1,file=fname,status="old")
    else
        open(1,file=fname,status="new")
    end if
    
    do j = 1,width
    do i = 1,height
        write(1,*) real(fig_out(i,j,:))/255
    end do
    end do
    
    close(1)
end program
