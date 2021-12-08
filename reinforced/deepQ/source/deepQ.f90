! deepQ
! By Prof. Carlo R. da Cunha, Ph.D.
! Last modification 02/12/2021
! creq@filosofisica.com

module deepQ_Mod
    use NobleNeuron
    implicit none
    
    integer,parameter,private   :: dp = kind(1.d0)    
    
    ! Action space
    integer,parameter   :: go_up    = 1
    integer,parameter   :: go_down  = 2
    integer,parameter   :: go_left  = 3
    integer,parameter   :: go_right = 4
    
    ! State space
    integer,parameter   :: state_A = 1
    integer,parameter   :: state_B = 2
    integer,parameter   :: state_C = 3
    integer,parameter   :: state_D = 4
    
    ! epsilon parameter
    real,parameter      :: eps = 0.1
    
    ! Game Class
    type,public :: game
        type(NeuralNetwork) :: Q
        real                :: alpha,gama
        real                :: replay_buffer(100,4)
        integer             :: rp_pointer
        integer             :: rp_top
        
        contains
        
        procedure,private   :: egreedy
        procedure           :: train
        procedure,private   :: add_replay
        procedure,private   :: trainNetwork
        procedure           :: print
    end type
    
    interface game
        procedure   :: game_constructor
    end interface
    
    contains
    
    !----------------------------!
    ! Constructor for class game !
    !----------------------------!
    function game_constructor(alpha,gama) result(self)
        implicit none
        
        real,intent(in) :: alpha,gama
        type(game)      :: self
        
        integer :: i
        
        self%alpha = alpha
        self%gama = gama        
        self%Q = NeuralNetwork(1,"L2","SGD")
        i = self%Q%addLayer(2,"tanh")
        i = self%Q%addLayer(3,"tanh")
        i = self%Q%addLayer(4,"tanh")
        i = self%Q%addLayer(6,"tanh")
        i = self%Q%addLayer(4,"tanh")
        
        self%rp_pointer = 1
        self%rp_top = 1
        self%replay_buffer = 0.0
    end function
    
    !-------------------------------!
    ! Add s,a,r,s' to replay buffer !
    !-------------------------------!
    subroutine add_replay(self,s,a,r,sl)
        implicit none
        
        class(game),intent(inout)   :: self
        integer,intent(in)          :: s,a,sl
        real,intent(in)             :: r
        
        self%replay_buffer(self%rp_pointer,:) = [real :: s,a,r,sl]
        self%rp_pointer = self%rp_pointer + 1
        if (self%rp_pointer > size(self%replay_buffer,1)) then
            self%rp_pointer = 1
        end if
        if (self%rp_top < 100) then
            self%rp_top = self%rp_top + 1
        end if
    end subroutine
    
    !--------------------------!
    ! Epsilon-greedy algorithm !
    !--------------------------!
    function egreedy(self,state) result(action)
        implicit none
        
        class(game),intent(inout)   :: self
        integer,intent(in)          :: state
        integer                     :: action
        
        real        :: r1,r2
        real        :: t(4,1)
        real(dp)    :: sample(1,1)
        integer     :: n(1)
        
        call random_number(r1)
        call random_number(r2)
        
        if (r1 < eps) then
            select case(state)
                case(state_A)
                    action = go_right
                case(state_B)
                    if (r2 < 0.5) then
                        action = go_left
                    else
                        action = go_down
                    end if
                case(state_C)
                    action = go_right
                case(state_D)
                    if (r2 < 0.5) then
                        action = go_left
                    else
                        action = go_up
                    end if
            end select
        else
            sample = reshape([real(dp) :: state],shape(sample))
            call self%Q%feedforward(sample)
            t = self%Q%output()
            n = maxloc(t(:,1))
            action = n(1)
        end if
    end function
    
    !----------------------------------------------!
    ! Play full episode until reach terminal state !
    !----------------------------------------------!
    subroutine train(self)
        implicit none
        
        class(game),intent(inout)   :: self
        
        real                :: r,x,Loss
        integer             :: i,j,s,a,sl,al,n,cnt
        integer,parameter   :: Tt = 2
        

        cnt = 1
        Loss = 1.0
        s = state_C
        do while (Loss > 1E-16)
            ! Choose random initial state s
            if (s .eq. state_C) then
                i = randint(1,3)
                select case(i)
                    case(1)
                        s = state_A
                    case(2)
                        s = state_B
                    case(3)
                        s = state_D
                end select
            end if
            
            ! Select action and collect reward for state s
            a = self%egreedy(s)
            ! Choose next state s' and add [s,a,r,s'] to replay buffer
            sl = next_state(s,a)
            r = reward(s,a)
            call self%add_replay(s,a,r,sl)
                            
            ! Every Tt, train network
            cnt = cnt + 1
            if (cnt > Tt) then
                cnt = 1
                x = self%trainNetwork()
                write(*,*) "Loss: ",x
                Loss = x
            end if
            
            ! Move to next state
            s = sl
       end do
    end subroutine
    
    ! Draw random integer between mi and ma
    function randint(mi,ma) result(x)
        implicit none
        
        integer,intent(in)  :: mi,ma
        integer             :: x
        real    :: r
        
        call random_number(r)
        x = mi + floor(r*(ma-mi+1))
    end function        
    
    ! Train network with minibatch
    function trainNetwork(self) result(L)
        implicit none
        
        class(game),intent(inout)   :: self
        real                        :: L
        
        integer,parameter   :: minibatch_size = 4
        integer     :: i,n,s,a,sl
        real        :: r,y,x
        real(dp)    :: Qsa
        real(dp)    :: sample(1,1)
        real(dp)    :: Q(4,1),gL(1,4)
        real(dp)    :: t(4,1)

        L = 0.0
        do n = 1,minibatch_size ! minibatch
            ! Select random entry in replay buffer
            i = randint(1,self%rp_top)
            s = int(self%replay_buffer(i,1))
            a = int(self%replay_buffer(i,2))
            r = self%replay_buffer(i,3)
            sl = int(self%replay_buffer(i,4))
            
            ! Get actions a' from state s'
            sample = reshape([real(dp) :: sl],shape(sample))
            call self%Q%feedforward(sample)
            Q = self%Q%output()
            
            ! Find y = r + gama.max Q(s',a')
            if (sl .eq. state_C) then
                y = r
            else
                y = r + self%gama*maxval(Q(:,1))
            end if
            
            ! Get actions from state s
            sample = reshape([real(dp) :: s],shape(sample))
            call self%Q%feedforward(sample)
            Q = self%Q%output()
            
            ! Loss = <(y-Q(s,a))^2>
            L = L + (y-Q(a,1))**2
                        
            ! Compute gradient from loss
            gL = 0.0
            gL(1,a) = Q(a,1)-y
            call self%Q%backpropagate(sample,gL)
        end do
        
        call self%Q%applyGrads(self%alpha)
        L = L/minibatch_size
    end function
    
    !-----------------!
    ! Find next state !
    !-----------------!
    function next_state(state,action) result(nstate)
        implicit none
        
        integer,intent(in)  :: state
        integer,intent(in)  :: action
        integer             :: nstate
        
        nstate = state
        select case(state)
            case(state_A)
                if (action .eq. go_right) then
                    nstate = state_B
                end if
            case(state_B)
                if (action .eq. go_left) then
                    nstate = state_A
                elseif (action .eq. go_down) then
                    nstate = state_D
                end if
            case(state_C)
                if (action .eq. go_right) then
                    nstate = state_D
                end if
            case(state_D)
                if (action .eq. go_left) then
                    nstate = state_C
                elseif (action .eq. go_up) then
                    nstate = state_B
                end if
        end select
    end function

    !-------------------------!
    ! Find reward for a state !
    !-------------------------!    
    function reward(state,action) result(r)
        implicit none
        
        integer,intent(in)  :: state,action
        real                :: r
        
       
        select case(state)
            case(state_A)
                select case(action)
                    case(go_right)
                        r = -0.01
                    case(go_left)
                        r = -0.1
                    case(go_up)
                        r = -0.1
                    case(go_down)
                        r = -0.1
                end select
            case(state_B)
                select case(action)
                    case(go_right)
                        r = -0.1
                    case(go_left)
                        r = -0.01
                    case(go_up)
                        r = -0.1
                    case(go_down)
                        r = -0.01
                end select
            case(state_C)
                r = 0.5
            case(state_D)
                select case(action)
                    case(go_right)
                        r = -0.1
                    case(go_left)
                        r = -0.01
                    case(go_up)
                        r = -0.01
                    case(go_down)
                        r = -0.1
                end select
        end select
    end function
    
    ! Print best strategy
    subroutine print(self)
        implicit none
        
        class(game),intent(inout)  :: self
        
        integer     :: i
        real(dp)    :: sample(1,1)
        real(dp)    :: t(4,1)
        integer     :: n(1)
        
        
        write(*,*) ""
        do i = 1,4
            select case(i)
                case(1)
                    write(*,*) "State A"
                case(2)
                    write(*,*) "State B"
                case(3)
                    write(*,*) "State C"
                case(4)
                    write(*,*) "State D"
            end select
            
            sample = reshape([real(dp) :: i],shape(sample))
            call self%Q%feedforward(sample)
            t = self%Q%output()
            n = maxloc(t(:,1))
            
            select case(n(1))
                case(1)
                    write(*,*) "-> go up"
                case(2)
                    write(*,*) "-> go down"
                case(3)
                    write(*,*) "-> go left"
                case(4)
                    write(*,*) "-> go right"
            end select
        end do
    end subroutine
        
end module    

!===============================================!
!                     MAIN                      !
!===============================================!
program test
    use deepQ_Mod
    implicit none
    
    type(Game)  :: G
    integer     :: i
    real        :: r
    
    G = Game(0.01,0.95)
    call G%train()
    call G%print()
    
end program
    
    
    
