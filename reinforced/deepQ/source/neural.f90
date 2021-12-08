
!**********************************************************************************************!
!                                   NEURAL NETWORK MODULE                                      !
!**********************************************************************************************!
module NobleNeuron
    use NNLayers
    
    implicit none
    
    integer,parameter,private   :: dp = kind(1.d0)    


    !------------------------------------------------------
    ! Neural Network Class
    !------------------------------------------------------
    type :: NeuralNetwork
        type(Layer),pointer :: head => null()
        type(Layer),pointer :: tail => null()
        
        integer             :: LayerNumber = 0
        integer             :: input_size
        
        character(len=20)   :: lossType
        character(len=20)   :: Optimizer
        
        ! Adam
        real(dp)            :: beta1,beta2,beta1n,beta2n
        
        contains
        
        ! Add Layers
        procedure           :: addLayer
        procedure,private   :: insertLayer
                
        ! Feedforward & backpropagation
        procedure           :: backpropagate
        procedure           :: feedforward
        procedure           :: applyGrads
        procedure           :: Loss, gradLoss
        procedure           :: output

    end type

    interface NeuralNetwork
        procedure   :: NN_Constructor
    end interface

    contains

    !------------------------------------------------------
    ! Neural Network Constructor
    !------------------------------------------------------
    function NN_Constructor(isize,loss,optimizer) result(self)
        implicit none
        
        type(NeuralNetwork) :: self
        integer,intent(in)              :: isize
        character(len=*),intent(in)     :: loss
        character(len=*),intent(in)     :: optimizer
        
    
        nullify(self%head)
        nullify(self%tail)
        
        self%LayerNumber = 0
        self%input_size = isize
        self%lossType = loss
        self%Optimizer = optimizer
        
        ! Adam
        self%beta1 = 0.9
        self%beta2 = 0.999
        self%beta1n = 0.9
        self%beta2n = 0.999
    end function

    !------------------------------------------------------
    ! Output
    !------------------------------------------------------
!     subroutine output(self,Z)
!         implicit none
!         
!         class(NeuralNetwork),intent(in) :: self
!         real(dp),intent(out)            :: Z(:,:)
!         
!         Z = self%tail%Z
!     end subroutine

    !------------------------------------------------------
    ! Feed Forward 
    !------------------------------------------------------
    subroutine feedforward(self,X)
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        real(dp),intent(in)                 :: X(:,:)
        
        class(Layer),pointer    :: tmp
        integer                 :: i
        
                
        tmp => self%head
        do while(associated(tmp))
            
            if (associated(tmp,self%head)) then
                call tmp%feed(X)
            else
                call tmp%feed(tmp%prev%Z)
            end if

            tmp => tmp%next
        end do
                
    end subroutine

    !------------------------------------------------------
    ! BackPropagation
    !------------------------------------------------------
    subroutine backpropagate(self,X,dL) 
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        real(dp),intent(in)                 :: X(:,:)
        real(dp),intent(in)                 :: dL(:,:)
    
        class(Layer),pointer    :: tmp
                
 
        tmp => self%tail
        do while(associated(tmp))
            !---------- TAIL ----------!
            if (associated(tmp,self%tail)) then
                !----- Binary X-Entropy -----!
                if (self%lossType .eq. "BCE" .or. self%lossType .eq. "REINFORCE") then
                    call tmp%back(tmp%prev%Z,dL,.true.)
                else
                    !--- Single Layer: Head = Tail ---!
                    if (associated(tmp,self%head)) then
                        call tmp%back(X,dL,.false.)
                    else
                    !--- Multilayer: Head != Tail ---!
                        call tmp%back(tmp%prev%Z,dL,.false.) 
                    end if
                end if
            !--- INTERMEDIATE LAYER ---!
            else
                !--- Head ---!
                if (associated(tmp,self%head)) then
                    call tmp%back(X,tmp%next%dL,.false.)
                else
                !--- Intermediate layer ---!
                    call tmp%back(tmp%prev%Z,tmp%next%dL,.false.)
                end if
                
            end if
        
            tmp => tmp%prev
        end do
    end subroutine    
    
    !======================================================
    ! OPTIMIZERS & GRADIENT
    !======================================================
    ! Update filters
    subroutine applyGrads(self,learning_rate)
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        real                                :: learning_rate
        
        class(Layer),pointer    :: tmp
        
        tmp => self%head
        
        ! Flatten, Pooling have no learnable parameters
        do while(associated(tmp))
            select case(self%Optimizer)
                case ("SGD")
                    call tmp%ApplyLayerGrads(learning_rate)
                case ("SGA")
                    call tmp%ApplyLayerAscent(learning_rate)
                case ("Adam")
                    call tmp%ApplyLayerAdam(learning_rate,self%beta1,self%beta2,self%beta1n,self%beta2n)
                    self%beta1n = self%beta1n*self%beta1
                    self%beta2n = self%beta2n*self%beta2
            end select
                    
            call tmp%ZeroLayerGrads()
            
            tmp => tmp%next
        end do
    end subroutine

    ! ZeroGradients
    subroutine ZeroGrads(self)
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        
        class(Layer),pointer    :: tmp
        
        tmp => self%head
        do while(associated(tmp))
            call tmp%ZeroLayerGrads()
            
            tmp => tmp%next
        end do
    end subroutine
        
    ! Output of neural network
    function output(self) result(Z)
        implicit none
        
        class(NeuralNetwork),intent(in) :: self
        real(dp)                        :: Z(size(self%tail%Z,1),size(self%tail%Z,2))
        
        Z = self%tail%Z
    end function
    
    ! Backpropagate error
    function dL(self) result(Z)
        implicit none
        
        class(NeuralNetwork),intent(in) :: self
        real(dp)                        :: Z(size(self%head%dL,1),size(self%head%dL,2))
        
        Z = self%head%dL
    end function
        
    !======================================================
    ! LOSS
    !======================================================
    function Loss(self,expected) result(L)
        use NNLoss
        implicit none
        
        class(NeuralNetwork),intent(in) :: self
        real(dp),intent(in)             :: expected(:,:)
        real(dp)                        :: L
        
        select case(self%lossType)
            case("BCE")
                L = sum(elementalLoss(self%tail%Y,expected,self%LossType))/size(self%tail%Y)
                
            case("REINFORCE")
                L = sum((expected - self%tail%Z)*self%tail%Y)
            
            case default
                L = sum(elementalLoss(self%tail%Z,expected,self%LossType))/size(self%tail%Z)
        
        end select
            
        
    end function
    
    ! Last gradient
    function GradLoss(self,expected) result(Z)
        use NNLoss
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        real(dp),intent(in)                 :: expected(:,:)
        real(dp)                            :: Z(size(expected,2),size(expected,1))
        
        
        select case(self%lossType)
            case("REINFORCE")
                Z = expected - self%tail%Z
            
            case default
                Z = deLoss(self%tail%Z,expected,self%lossType)
        end select
    end function    
        
    !======================================================
    ! ADD LAYERS
    !======================================================
    
    ! Add Generic Layer
    function insertLayer(self,activation) result(no)
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        character(len=*),intent(in)         :: activation
        integer                             :: no
                
        class(Layer),pointer    :: tmp
        
        
        self%LayerNumber = self%LayerNumber + 1
        no = self%LayerNumber
        
        allocate(tmp)
        tmp%layerNumber = no
        tmp%activation = activation
        
        ! Link layer
        tmp%next => null()
        if (.not. associated(self%head)) then
            tmp%prev => null()
            tmp%isize = self%input_size
            self%head => tmp
            self%tail => tmp
        else
            tmp%prev => self%tail
            tmp%isize = size(self%tail%Z)
            self%tail%next => tmp
            self%tail => tmp
        end if
    end function

   
    
    ! Add Dense Layer
    function addLayer(self,output_size,activation) result(no)
        implicit none
        
        class(NeuralNetwork),intent(inout)  :: self
        integer,intent(in)                  :: output_size
        character(len=*),intent(in)         :: activation
        integer                             :: no
        
        ! Allocate
        no = self%insertLayer(activation)
        allocate(self%tail%W(output_size,self%tail%isize))
        allocate(self%tail%dW(self%tail%isize,output_size))
        allocate(self%tail%B(output_size,1))
        allocate(self%tail%dB(1,output_size))
        allocate(self%tail%Y(output_size,1))
        allocate(self%tail%Z(output_size,1))
        allocate(self%tail%dL(1,self%tail%isize))
        ! Adam
        allocate(self%tail%Wm(self%tail%isize,output_size))
        allocate(self%tail%Wv(self%tail%isize,output_size))
        allocate(self%tail%Bm(1,output_size))
        allocate(self%tail%Bv(1,output_size))
                
        ! Zero gradients
        self%tail%dW = 0.0
        self%tail%dB = 0.0
        self%tail%Wm = 0.0
        self%tail%Wv = 0.0
        self%tail%Bm = 0.0
        self%tail%Bv = 0.0
        
        ! Randomize filters
        call random_number(self%tail%W)
        self%tail%W = 2*(self%tail%W - 0.5)/sqrt(real(self%tail%isize))
        self%tail%B = 0.0   
                           
        write(*,*) "Added Dense layer,       #",no,"in :",self%tail%isize
        write(*,*) "                         #",no,"out:",shape(self%tail%Z),self%tail%activation
    end function   
    
end module
