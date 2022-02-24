#include <NobleNeuron>
#include <tenten>

#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

/****************************************
 * Choose an action based on the policy *
 * **************************************/
int choose(float pi[9]) {
    float   cum[9];
    int     a;
    
    // Cumulative
    cum[0] = pi[0];
    for(a = 1; a < 9; a ++)
        cum[a] = pi[a] + cum[a-1];
    
    // Normalize
    for(a = 0; a < 9; a ++) 
        cum[a] /= cum[8];
                    
    // Choose action
    float r = (float)rand()/RAND_MAX;
    a = 0;
    while(r > cum[a])
        a ++;
    
    return a;
}

/*********************
 * Move to new state *
 * *******************/
void moveto(int a, int &x, int &y, int BoardSize) {
    switch(a) {
        case 0:
            x ++;
            y ++;
            break;
        case 1:
            x ++;
            break;
        case 2:
            x ++;
            y --;
            break;
        case 3:
            y ++;
            break;
        case 5:
            y --;
            break;
        case 6:
            x --;
            y ++;
            break;
        case 7:
            x --;
            break;
        case 8:
            x --;
            y --;
            break;
    }
    if (x >= BoardSize)
        x = 0;
    if (x < 0)
        x = BoardSize - 1;
    if (y >= BoardSize)
        y = 0;
    if (y < 0)
        y = BoardSize - 1;
}

/* ***************
 * MAIN FUNCTION *
 * ***************/
int main(void) {
    // Declaration of Constants
    const int           BoardSize = 3;
    const int           MaxSteps = 150;
    const unsigned int  xo = 1;
    const unsigned int  yo = 1;
    const float         gamma = 0.8;
    
    // Declaration of Variables
    bool    active;
    int     a,T;
    int     t;
    float   R,Loss;
    float   reward;
        
    int     x,y;
    
    // Neural net
    FeedForward Actor(2);
    tensor      pi(9,1);
    tensor      g(1,9), grad[MaxSteps];
    tensor      input(2,1);
    
    ofstream myfile;
        
    //--------- Code ----------
    myfile.open("out.dat");
    
    // Setup feedforward neural network
    Actor.add_layer(4,"sigmoid");
    Actor.add_layer(6,"sigmoid");
    Actor.add_layer(9,"none");

    // Initialize tensors with proper size
    for(unsigned int i = 0; i < MaxSteps; i ++)
        grad[i].set(1,9);    
    
    // Initialization
    srand(time(NULL));
        
    // Main loop
    do {
        // Get an average loss
        Loss = 0;
        for(int avg = 0; avg < 10; avg ++) {
            /*---------------------
             *  Sample trajectory
             * --------------------*/
            t = 0;
            active = true;
            x = rand()%BoardSize;
            y = rand()%BoardSize;
            R = 0;
            do {
                // Run policy
                input = {float(x),float(y)};
                Actor.feedforward(input);
                pi = Actor.output().softmax();
                a = pi.choose();
                
                // Store trajectory
                grad[t] = pi.REINFORCE_grad_loss(a);
                
                // Move to new state
                moveto(a,x,y,BoardSize);                
                
                // Obtain reward
                reward = (x == xo && y == yo) ? 100 : -1;
                R += pow(gamma,t)*reward;
                                
                t ++;
                if (t >= MaxSteps || (x == xo && y == yo))
                    active = false;
                
            } while(active); 
            T = t;
            cout << "Ended episode after " << T << " steps" << endl;
            Loss += T;

            // Gradient ascent            
            g.zeros();
            for(t = 0; t < T; t ++) {
                g += grad[t]*R;
            }
            Actor.backpropagate(g); 
        }
        Actor.applygrads(-0.001);
        Loss /= 10;
        myfile << Loss << endl;
    } while(Loss > 1);
    
    myfile.close();
}
            
    
