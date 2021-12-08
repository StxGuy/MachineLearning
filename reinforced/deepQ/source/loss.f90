module NNLoss

    integer,parameter,private   :: dp = kind(1.d0)    
    
    contains


    !=================================================================================
    ! ELEMENTAL LOSS FUNCTIONS
    !=================================================================================
    ! Elemental loss
    ! a: estimated, b: expected
    elemental function elementalLoss(estimated,expected,losstype) result(c)
        implicit none
        
        real(dp),intent(in)         :: estimated,expected
        character(len=*),intent(in) :: losstype
        real(dp)                    :: c
        
        select case(losstype)
            case("L1")
                c = abs(expected-estimated)
                
            case("L2")
                c = (expected-estimated)**2
                
            case("BCE")
                c = maxval([real(dp) :: 0,estimated]) - expected*estimated + log(1.0+exp(-abs(estimated)))
        end select
    end function    
    
    
    ! Elemental derivative of the loss function
    ! a: estimated, b: expected
    elemental function elementalDeLoss(estimated,expected,losstype) result(c)
        implicit none
        
        real(dp),intent(in)         :: estimated,expected
        character(len=*),intent(in) :: losstype
        real(dp)                    :: c
        
        real(dp),parameter          :: threshold = 1E-7
        
        select case(losstype)
            case("L1")
                c = sign(1.d0,estimated - expected)
            case("L2")
                c = 2*(estimated - expected)
            case("BCE")
                c = estimated - expected
                    
        end select
    end function
    
    !=================================================================================
    ! TENSOR FUNCTIONS
    !=================================================================================
    ! Gradient of loss function
    function deLoss(estimated,expected,losstype) result(r)
        implicit none
        
        real(dp),intent(in)     :: estimated(:,:)
        real(dp),intent(in)     :: expected(:,:)
        character(len=*)        :: losstype
        real(dp)                :: r(size(estimated,2),size(estimated,1))
        
        integer     :: i
        real(dp)    :: t(size(estimated,1),size(estimated,2))
        
        
        t = elementalDeLoss(estimated,expected,losstype)
        r = transpose(t)
    end function
    
end module
