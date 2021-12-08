module NNActiv
    implicit none
    
    integer,parameter,private   :: dp = kind(1.d0)
    
    contains

    !-- Elemental Activation --!
    elemental function elementalActivation(y,activation) result(z)
        implicit none
        
        real(dp),intent(in)         :: y
        character(len=*),intent(in) :: activation
        real(dp)                    :: z
        
        select case(activation)
            case("Sigmoid")
                if (y > 100.0) then
                    z = 1.0/(1.0 + exp(-100.0))
                else if (y < -100.0) then
                    z = exp(-100.0)
                else
                    z = 1.0/(1.0 + exp(-y))
                end if
            
            case("ReLU")
                if (y < 0.0) then
                    z = 0.0
                else
                    z = y
                end if
                
            case("lReLU")
                if (y < 0.0) then
                    z = 1E-3*y
                else
                    z = y
                end if
                
            case("tanh")
                if (y > 100.0) then
                    z = tanh(100.0)
                else if (y < -100.0) then
                    z = tanh(-100.0)
                else
                    z = tanh(y)
                end if
                
            case("soft+")
                if (y > 100.0) then
                    z = y
                else
                    z = log(1.0+exp(y))
                end if
                
            case("none")
                z = y
            
        end select
    end function
    
    !-- Elemental Deactivation --!
    elemental function elementalDeactivation(y,z,activation) result(dzdy)
        implicit none
        
        real(dp),intent(in)         :: y,z
        character(len=*),intent(in) :: activation
        real(dp)                    :: dzdy
        
        select case(activation)
            case("Sigmoid")
                dzdy = z*(1-z)
                
            case("ReLU")
                if (y > 0) then
                    dzdy = 1.0
                else
                    dzdy = 0.0
                end if
                
            case("lReLU")
                if (y > 0) then
                    dzdy = 1.0
                else
                    dzdy = 1E-3
                end if
                
            case("tanh")
                dzdy = 1-z*z
                
            case("soft+")
                dzdy = 1-exp(-z)
                                
            case("none")
                dzdy = y
            
        end select
    end function         
    
    !-- ACTIVATE --!
    function activate(Y,activation) result(Z)
        implicit none
        
        real(dp),intent(in)         :: Y(:,:)
        character(len=*),intent(in) :: activation
        real(dp)                    :: Z(size(Y,1),size(Y,2))
        
        if (activation .eq. "softmax") then
            Z = softmax(Y)        
        else
            Z = elementalActivation(Y,activation)
        end if
    end function
        
        
    
    !-- DEACTIVATE --!
    function deactivate(dLdZ,Y,Z,activation) result(dLdY)
        implicit none
        
        real(dp),intent(in)         :: dLdZ(:,:)
        real(dp),intent(in)         :: Y(:,:)
        real(dp),intent(in)         :: Z(:,:)
        character(len=*),intent(in) :: activation
        real(dp)                    :: dLdY(size(dLdZ,1),size(dLdZ,2))
        
        integer     :: i,j
               
        
        do j = 1,size(dLdY,2)
            do i = 1,size(dLdY,1)
                if (activation .eq. "softMax") then
                    dLdY(i,j) = -Z(j,i)*Z(j,i)
                    if (i == j) then
                        dLdY(j,j) = dLdY(j,j) + Z(j,i)
                    end if
                    dLdY(i,j) = dLdY(i,j)*dLdZ(i,j)
                else
                    dLdY(i,j) = dLdZ(i,j)*elementalDeactivation(Y(j,i),Z(j,i),activation)
                end if
            end do
        end do

            
    end function
    
    !-- SOFTMAX --!
    function softmax(Y) result(Z)
        implicit none
        
        real(dp),intent(in) :: Y(:,:)
        real(dp)            :: Z(size(Y,1),size(Y,2))
        
        real(dp)    :: mx,s
        integer     :: i
        

        mx = maxval(Y(:,1))
        
        s = 0
        do i = 1,size(Y,1)
            Z(i,1) = exp(Y(i,1)-mx)
            s = s + Z(i,1)
        end do
        do i = 1,size(y,1)
            Z(i,1) = Z(i,1)/s
        end do
    end function
            
    
end module
