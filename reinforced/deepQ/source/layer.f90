!**********************************************************************************************!
!                                        LAYER MODULE                                          !
!**********************************************************************************************!
module NNLayers
    implicit none
    
    integer,parameter,private   :: dp = kind(1.d0)    
    
    !------------------------------------------------------
    ! Layer Class
    !------------------------------------------------------
    type :: Layer
        !---------- General Parameters -----------!
        character(len=20)       :: activation
        integer                 :: layerNumber
        integer                 :: isize

        !---------- Specific Parameters -----------!
        real(dp),allocatable    :: W(:,:), dW(:,:)
        real(dp),allocatable    :: B(:,:), dB(:,:)
        
        !----------- Input & Output --------------!
        real(dp),allocatable    :: Z(:,:)
        real(dp),allocatable    :: Y(:,:)     
        real(dp),allocatable    :: dL(:,:)
        
        !---------------- Adam -------------------!
        real(dp),allocatable    :: Wm(:,:), Wv(:,:)
        real(dp),allocatable    :: Bm(:,:), Bv(:,:)
        
        !------------ Linked List ----------------!
        type(Layer),pointer     :: next,prev
        
        contains
        
        procedure               :: feed,back
        procedure               :: ZeroLayerGrads
        procedure               :: ApplyLayerGrads
        procedure               :: ApplyLayerAscent
        procedure               :: ApplyLayerAdam
        final                   :: LayerDestructor
    end type
    
    interface Layer
        procedure   :: LayerConstructor
    end interface
    
    
    contains
    
    !-------------------------!
    !       CONSTRUCTOR       !
    !-------------------------!
    function LayerConstructor(layerNumber,sizeIn,activation) result(self)
        implicit none
        
        integer,intent(in)          :: layerNumber
        integer,intent(in)          :: sizeIn
        character(len=*),intent(in) :: activation
        
        type(Layer) :: self
        
        self%layerNumber = layerNumber
        self%isize = sizeIn
        self%activation = activation
        
        self%next => null()
        self%prev => null()
    end function
    
    !-------------------------!
    !       DESTRUCTOR        !
    !-------------------------!
    subroutine LayerDestructor(self)
        implicit none
        
        type(Layer) :: self
        
        if (allocated(self%W)) then
            deallocate(self%W)
        end if
        if (allocated(self%dW)) then
            deallocate(self%dW)
        end if
        if (allocated(self%B)) then
            deallocate(self%B)
        end if
        if (allocated(self%dB)) then
            deallocate(self%dB)
        end if
        if (allocated(self%Z)) then
            deallocate(self%Z)
        end if
        if (allocated(self%Y)) then
            deallocate(self%Y)
        end if
        if (allocated(self%dL)) then
            deallocate(self%dL)
        end if
    end subroutine             
    
    
    !------------------------------------------------------
    ! Feed Forward
    !------------------------------------------------------
    subroutine feed(self,X) 
        use NNActiv
        implicit none
        
        class(Layer),intent(inout)  :: self
        real(dp),intent(in)         :: X(:,:)
        
        integer     :: i

        self%Y = matmul(self%W,X) + self%B
        self%Z = Activate(self%Y,self%activation)
    end subroutine
    
    !------------------------------------------------------
    ! Backpropagation
    !------------------------------------------------------
    subroutine back(self,X,dLdZ,skip_activation) 
        use NNActiv
        implicit none
        
        class(Layer),intent(inout)  :: self
        real(dp),intent(in)         :: X(:,:)
        real(dp),intent(in)         :: dLdZ(:,:)
        logical,intent(in)          :: skip_activation
        
        real(dp)    :: dLdY(size(dLdZ,1),size(dLdZ,2))
        
        integer     :: k
        real        :: r
                

        ! Skip activation ?
        if (skip_activation .or. self%activation .eq. "none") then
            dLdY = dLdZ
        else
            dLdY = deactivate(dLdZ,self%Y,self%Z,self%activation)
        end if
        
        ! Backpropagate
        self%dB = self%dB + dLdY
        self%dW = self%dW + matmul(X,dLdY)
        self%dL = matmul(dLdY,self%W)
    end subroutine

    !------------------------------------------------------
    ! Zero Gradients
    !------------------------------------------------------
    subroutine ZeroLayerGrads(self)
        implicit none
        
        class(Layer),intent(inout)  :: self
        

        if (allocated(self%dW)) then
            self%dW = 0.0
        end if
        if (allocated(self%dB)) then
            self%dB = 0.0
        end if
    end subroutine
    
    !------------------------------------------------------
    ! Apply Gradients
    !------------------------------------------------------
    subroutine ApplyLayerGrads(self,lrate)
        implicit none
        
        class(Layer),intent(inout)  :: self
        real,intent(in)             :: lrate
        

        if (allocated(self%dW)) then
            self%W = self%W - lrate*transpose(self%dW)
        end if
        if (allocated(self%dB)) then
            self%B = self%B - lrate*transpose(self%dB)
        end if
    end subroutine
    
    subroutine ApplyLayerAscent(self,lrate)
        implicit none
        
        class(Layer),intent(inout)  :: self
        real,intent(in)             :: lrate
        
        
        if (allocated(self%dW)) then
            self%W = self%W + lrate*transpose(self%dW)
        end if
        if (allocated(self%dB)) then
            self%B = self%B + lrate*transpose(self%dB)
        end if
    end subroutine
        
    !------------------------------------------------------
    ! Adam
    !------------------------------------------------------
    subroutine ApplyLayerAdam(self,lrate,beta1,beta2,beta1n,beta2n)
        implicit none
        
        class(Layer),intent(inout)  :: self
        real,intent(in)             :: lrate
        real(dp),intent(in)         :: beta1,beta2,beta1n,beta2n
        
        real,parameter      :: esp = 1.0E-8
        real,parameter      :: tau = 10
        real                :: r
        integer             :: k
        
        real(dp),allocatable    :: bh(:,:),bv(:,:)
        real(dp),allocatable    :: wh(:,:),wv(:,:)
        
        

        if (allocated(self%dW)) then
            self%Wm = beta1*self%Wm + (1-beta1)*self%dW
            self%Wv = beta2*self%Wv + (1-beta2)*self%dW*self%dW
            
            allocate(wh(size(self%dW,1),size(self%dW,2)))
            allocate(wv(size(self%dW,1),size(self%dW,2)))
            
            wh = self%Wm/(1-beta1n)
            wv = self%Wv/(1-beta2n)
            
            ! Clipping
            wh = wh/(sqrt(wv)+esp)
            r = sqrt(sum(wh*wh))
            if (r > tau) then
                wh = tau*wh/r
            end if
            
            self%W = self%W - lrate*wh
            
            deallocate(wh,wv)
        end if
        
        ! BIAS
        if (allocated(self%dB)) then
            self%Bm = beta1*self%Bm + (1-beta1)*self%dB
            self%Bv = beta2*self%Bv + (1-beta2)*self%dB*self%dB
            
            allocate(bh(size(self%dB,1),size(self%dB,2)))
            allocate(bv(size(self%dB,1),size(self%dB,2)))
            
            bh = self%Bm/(1-beta1n)
            bv = self%Bv/(1-beta2n)
            
            ! Clipping
            bh = bh/(sqrt(bv)+esp)
            r = sqrt(sum(bh*bh))
            if (r > tau) then
                bh = tau*bh/r
            end if
            
            self%B(:,:) = self%B(:,:) - lrate*transpose(bh(:,:))
      
            deallocate(bh,bv)
        end if
    end subroutine
    
end module
