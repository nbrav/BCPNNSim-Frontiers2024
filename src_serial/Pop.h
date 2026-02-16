/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#ifndef __Pop_included
#define __Pop_included

#include <vector>
#include <string>
#include <random>
#include <cfloat>
#include <list>

#include "Globals.h"
#include "Prj.h"

class Pop {

 public:

    static int npop;
    int id;
    long int H, M, N;    
    std::string actfn;
    float eps, dr;    
    float *sup, *act;
    std::vector<float> energyhistory;

    Pop(long int H, long int M, std::string actfn = "softWTA");
    ~Pop();

    // I/O functions
    void store(std::string field, FILE*);

    // Core functions
    void updact();
    void propagate(float*, Prj*);
    void injectNoise(float nampl);
    void execone(float* Xs, Prj* prj, float nampl = 0.001, int niter = 1, bool monitoring = false);

private:

} ;

#endif // __Pop_included
