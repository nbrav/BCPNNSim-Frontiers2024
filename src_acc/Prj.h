/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#ifndef __Prj_included
#define __Prj_included

#include <vector>
#include <string>
#include <random>
#include <cfloat>
#include <list>

#include "Globals.h"

#ifdef _CUDA
#include <curand.h>
#include <cublas_v2.h>
#include <openacc.h> // for acc_get_cuda_stream
#endif

class Prj {

 public:

    static int nprj;

    int id;

    long int Hs, Ms, Ns, Hr, Mr, Nr, Nrs, Hrs;

    // synaptic plasticity parameters
    std::string lrule;
    float eps, C, lr;
    float *Cs, *Cr, *Crs, *Wrs, *Br;

    // structural plasticity parameters
    int *WConnrs, *Connrs;
    float *mutual_info, *score;
    int nconn;

    Prj(int Hs,int Ms,int Hr,int Mr,std::string lrule = "BCPNN");

    ~Prj();

    void initmemtraces();

    void store(std::string field, FILE*);

    void loadconn(std::string); 

    void set_learningrate(float);
  
    void trainone(float* Xs, float* Xr);

    void updbw();

    void initconn(int);

    void updconn();

    void updwconn();

    int updconn_nswap = 100;

    float updconn_threshold = 1.1;
    
#ifdef _OPENACC

    float plr, pC;
    float* restrict pCs;
    float* restrict pCr;
    float* restrict pCrs;

    float* restrict pBr;
    float* restrict pWrs;

    int* restrict pConnrs;
    int* restrict pWConnrs;
    float* restrict pscore;
    float* restrict pmutual_info;

    void trainone_acc(float* restrict pXs, float* restrict pXr);

    void updbw_acc();

    void updconn_acc();

    void updwconn_acc();
    
#endif

private:

} ;

#endif // __Prj_included
