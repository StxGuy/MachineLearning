! SARSA
! By Prof. Carlo R. da Cunha, Ph.D.
! Last modification 02/12/2021
! creq@filosofisica.com

module SARSA_Mod
    implicit none
    
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
        real    :: Q(4,4)
        real    :: alpha,gama
        
        contains
        
        procedure,private   :: egreedy
        procedure           :: play_episode
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
        
        self%alpha = alpha
        self%gama = gama        
        self%Q = 0.0
    end function
    
    !--------------------------!
    ! Epsilon-greedy algorithm !
    !--------------------------!
    function egreedy(self,state) result(action)
        implicit none
        
        class(game),intent(in)  :: self
        integer,intent(in)      :: state
        integer                 :: action
        
        real    :: r1,r2
        integer :: n(1)
        
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
            n = maxloc(self%Q(state,:))
            action = n(1)
        end if
    end function
    
    !----------------------------------------------!
    ! Play full episode until reach terminal state !
    !----------------------------------------------!
    function play_episode(self) result(cR)
        implicit none
        
        class(game),intent(inout)   :: self
        real                        :: cR
        
        real    :: r
        integer :: i,s,a,sl,al
        
        ! Choose random initial state s
        call random_number(r)
        i = floor(r*3)
        select case(i)
            case(0)
                s = state_A
            case(1)
                s = state_B
            case(2)
                s = state_D
        end select
        
        ! Find corresponding a epsilon-greedy
        a = self%egreedy(s)
        
        ! Run policy until terminal state
        cR = 0.0
        do while(s .ne. state_C)
            r = reward(s)
            sl = next_state(s,a)
            al = self%egreedy(sl)
            
            self%Q(s,a) = (1-self%alpha)*self%Q(s,a) + self%alpha*(r+self%gama*self%Q(sl,al))
            s = sl
            a = al
            cR = cR + r
        end do
        cR = cR + reward(state_C)
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
    function reward(state) result(r)
        implicit none
        
        integer,intent(in)  :: state
        real                :: r
        
        select case(state)
            case(state_A)
                r = -1.0
            case(state_B)
                r = -0.5
            case(state_C)
                r = 10.0
            case(state_D)
                r = -0.25
        end select
    end function
        
end module    

!===============================================!
!                     MAIN                      !
!===============================================!
program test
    use SARSA_Mod
    implicit none
    
    type(Game)  :: G
    integer     :: i
    
    G = Game(0.70,0.95)
    
    do i = 1,30
        write(*,*) G%play_episode()
    end do
    
end program
    
    
    
