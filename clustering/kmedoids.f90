
module kmod
    
    contains

    ! Metric distance between two vectors
    ! let's use Euclidean metric
    function distance(a,b) result(c)
        implicit none
        
        real,intent(in) :: a(:),b(:)
        real            :: c
        
        integer :: i
        
        c = 0
        do i = 1,size(a,1)
            c = c + (a(i)-b(i))**2
        end do
        c = sqrt(c)
    end function

    function cost(medoids,dados,labels) result(c)
        implicit none
        
        real,intent(in)     :: medoids(:,:)
        real,intent(in)     :: dados(:,:)
        integer,intent(in)  :: labels(:)
        real                :: c
        
        integer :: i
                
        do i = 1,size(dados,1)
            c = c + distance(dados(i,:),medoids(labels(i),:))
        end do
    end function
end module

program kmed
    use kmod
    implicit none
    
    integer,parameter   :: datalength = 300
    real                :: data(datalength,2)
    real                :: medoid(3,2)
    real                :: d,d_min
    real                :: c1,c2
    real                :: tmp(2),best(2)
    integer             :: i,j,k(1),swap
    integer             :: label(datalength)
    real                :: s(datalength)
    integer             :: clustersize(3)
    real                :: t(3)
    real                :: a,b
    
    character(len=50)   :: fname
    logical             :: existent
    logical             :: change
    
    
    
    ! Load image
    open(1,file="data.dat",status="old")
    
    do i = 1,datalength
        read(1,*) data(i,:)
    end do
    
    close(1)    
    
    ! Set itial medoids
    medoid(1,:) = data(50,:)
    medoid(2,:) = data(150,:)
    medoid(3,:) = data(250,:)
    
    swap = 1
    do while(swap > 0)
        ! Assignment step
        do i = 1,datalength
            d_min = 1E5
            do j = 1,3
                d = distance(data(i,:),medoid(j,:))
                if (d < d_min) then
                    label(i) = j
                    d_min = d
                end if
            end do
        end do
        
        ! For each medoid...
        swap = 0
        do j = 1,3
            c1 = cost(medoid,data,label)
            tmp = medoid(j,:)
            ! ...check if non-medoid point has lower cost, and...
            do i = 1,datalength
                medoid(j,:) = data(i,:)
                c2 = cost(medoid,data,label)
                ! ... remember the best choice.
                if (c2 < c1) then
                    c1 = c2
                    best = medoid(j,:)
                    change = .true.
                end if
            end do
            ! If any non-medoid improved, swap
            if (change) then
                medoid(j,:) = best
                swap = swap + 1
            else
                medoid(j,:) = tmp
            end if
        end do
    end do
    
    ! Save labels
    fname = "labels.dat"
    inquire(file=fname,exist=existent)
    
    if (existent) then
        open(1,file=fname,status="old")
    else
        open(1,file=fname,status="new")
    end if
    
    do i = 1,datalength
        write(1,*) label(i)
    end do
    
    close(1)
    do i = 1,3
        write(*,*) medoid(i,:)
    end do
    
    ! Silhouette
    ! find cluster sizes
    clustersize = 0
    s = 0
    t = 0
    do i = 1,datalength
        do j = 1,3
            if (label(i) == j) then
                clustersize(j) = clustersize(j) + 1
            end if
        end do
    end do
    
    ! Find coefficients a and b
    do i = 1,datalength
        a = 0
        d_min = 1E5
        do j = 1,datalength
            ! If they are from the same cluster
            if (label(j) .eq. label(i)) then
                if (i .ne. j) then
                    a = a + distance(data(i,:),data(j,:))
                end if
            else
                t(label(j)) = t(label(j)) + distance(data(i,:),data(j,:))
            end if
        end do
        if (clustersize(label(i)) .le. 1) then
            s(i) = 0
        else
            a = a/(clustersize(label(i))-1)
            
            select case(label(i))
                case(1)
                    if (t(2) < t(3)) then
                        b = t(2)/clustersize(2)
                    else
                        b = t(3)/clustersize(3)
                    end if
                case(2)
                    if (t(1) < t(3)) then
                        b = t(1)/clustersize(1)
                    else
                        b = t(3)/clustersize(3)
                    end if
                case(3)
                    if (t(1) < t(2)) then
                        b = t(1)/clustersize(1)
                    else
                        b = t(2)/clustersize(2)
                    end if
            end select            
                        
            write(*,*) a,b,a/b
            
            if (a < b) then
                s(i) = 1.0-a/b
            end if
            if (a == b) then
                s(i) = 0
            end if
            if (a > b) then
                s(i) = b/a-1.0
            end if
        end if
    end do
    
    ! Save filhouette
    fname = "silhouette.dat"
    inquire(file=fname,exist=existent)
    
    if (existent) then
        open(1,file=fname,status="old")
    else
        open(1,file=fname,status="new")
    end if
    
    do i = 1,datalength
        write(1,*) s(i)
    end do
    
    close(1)        
                
    
end program
