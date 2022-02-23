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
    const float         eta = 0.001;
    
    // Declaration of Variables
    bool    active;
    int     a,T;
    int     t;
    float   R,Loss;
    float   reward;
    float   delta;
    float   pi[9];
    float   s1[MaxSteps],s2[MaxSteps];
    int     action[MaxSteps];
    
    
    int x,y;
    float mu,w1,w2,w3;
    
    
    ofstream myfile;
        
    //--------- Code ----------
    myfile.open("out.dat");
    
    // Initialization
    srand(time(NULL));
    w1 = (float)rand()/RAND_MAX-0.5;
    w2 = (float)rand()/RAND_MAX-0.5;
    w3 = (float)rand()/RAND_MAX-0.5;
        
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
                mu = w1*x + w2*y + w3;
                for(a = 0; a < 9; a ++) 
                    pi[a] = exp(-pow(a-mu,2));
                
                a = choose(pi);                
                
                // Store trajectory
                s1[t] = x;
                s2[t] = y;
                action[t] = a;
                
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
            for(t = 0; t < T; t ++) {
                delta = action[t] - w1*s1[t] - w2*s2[t] - w3;
                w1 += eta*delta*s1[t]*R/T;
                w2 += eta*delta*s2[t]*R/T;
                w3 += eta*delta*R/T;
            }
        }
        Loss /= 10;
        myfile << Loss << endl;
    } while(Loss > 1);
    
    myfile.close();
    cout << "Model: " << w1 << "," << w2 << "," << w3 << endl;        
}
            
    
