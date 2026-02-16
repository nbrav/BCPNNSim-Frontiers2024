/*

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2021-08-02     Modified: 2021-12-06

*/

#include <vector>
#include <string>
#include <random>

#include "Globals.h"
#include "Pop.h"

#include <iostream>

using namespace std;
using namespace Globals;

int Pop::npop = 0;

Pop::Pop(long int H, long int M, string actfn) {

    id = npop++;

    this->H = H;
    this->M = M;

    N = H * M;
    
    this->actfn = actfn;

    eps = 1e-7;
        
    sup = new float[N];
    act = new float[N];

    psup = new float[N];
    for (long int n=0; n<N; n++) psup[n] = 0;

    pact = new float[N];
    for (long int n=0; n<N; n++) pact[n] = 0;

#pragma acc enter data create(this)

#pragma acc update device(this)

#pragma acc enter data copyin(psup[:N], pact[:N])
    
#ifdef _CUDA
    
    istat = curandCreateGenerator(&gt, CURAND_RNG_PSEUDO_DEFAULT);
    
    if (istat != CURAND_STATUS_SUCCESS)
        cerr << "CURAND initialization failed" << endl;

    istat = curandSetStream(gt, (cudaStream_t)acc_get_cuda_stream(1));
	
    pGnextfloat = new float[N]();    

#pragma acc enter data copyin(pGnextfloat[:N])

#endif

}

Pop::~Pop() {
    
#pragma acc exit data delete(psup[:N], pact[:N])
    
    delete[] psup, pact;
    
}

void Pop::store(std::string field, FILE* f) {

    if (field == "act") {
        
#pragma acc update host(pact[:N]) 

        fwrite(pact, sizeof(float), N, f);    
    
    } else if (field == "sup") {

#pragma acc update host(psup[:N]) 

        fwrite(psup, sizeof(float), N, f);    

    } else
        
        printf("\nPop::store Invalid field!");
            
}

float* Pop::getact() {

#ifdef _OPENACC
    return pact;
#else
    return act;
#endif
    
}

void Pop::updact() {

    if (actfn=="WTA")

        error("Pop::updact","actfn=='WTA' not yet implemented");
		
    else if (actfn=="kWTA")
		
        error("Pop::updact","actfn=='kWTA' not yet implemented");
	
    else if (actfn=="softWTA") {

#pragma acc data present(psup[:N], pact[:N])
#pragma acc parallel loop async(1)
        for (long int h=0; h<H; h++) {
            
            float supmax = -FLT_MAX, esupsum = 0;
#pragma acc loop reduction(max:supmax)
            for (long int m=0; m<M; m++) supmax = max(supmax, psup[M*h+m]);
#pragma acc loop 	    
            for (long int m=0; m<M; m++) pact[M*h+m] = exp(psup[M*h+m]-supmax);
#pragma acc loop reduction(+:esupsum)	    
            for (long int m=0; m<M; m++) esupsum += pact[M*h+m];
#pragma acc loop 	    
            for (long int m=0; m<M; m++) pact[M*h+m] /= esupsum;
            
        }
                
    } else if (actfn=="stochWTA") {

#pragma acc host_data use_device(pGnextfloat)
    
        istat = curandGenerateUniform(gt, pGnextfloat, H);
        curandGenerateUniform(gt, pGnextfloat, H);

#pragma acc data present(psup[:N], pact[:N])
#pragma acc parallel loop async(1)
        for (int h=0; h<H; h++) {
            
            float supmax = -FLT_MAX, esupsum = 0;
#pragma acc loop reduction(max:supmax)
            for (int m=0; m<M; m++) supmax = max(supmax, psup[M*h+m]);
#pragma acc loop 	    
            for (int m=0; m<M; m++) pact[M*h+m] = exp(psup[M*h+m]-supmax);
#pragma acc loop reduction(+:esupsum)	    
            for (int m=0; m<M; m++) esupsum += pact[M*h+m];
#pragma acc loop 	    
            for (int m=0; m<M; m++) pact[M*h+m] /= esupsum;
            // Categorical sampling way
#pragma acc loop // cumulative sum, is this okay with data dependencies inside?
            for (int m=1; m<M; m++) pact[M*h+m] += pact[M*h+m-1];
            float rand = pGnextfloat[h];
#pragma acc loop // choosing one category
            for (int m=0; m<M; m++) pact[M*h+m] = (rand<=pact[M*h+m]) and (rand>pact[M*h+m-1]);
            // Independent sampling way
// #pragma acc loop
//            for (int m=0; m<M; m++) pact[M*h+m] = pGnextfloat[M*h+m] < pact[M*h+m];
                        
        }
        
    } else if (actfn=="HALF")
	
        error("Pop::updact","actfn=='HALF' not yet implemented");
	
    else
		
        error("Pop::updact","No such actfn: " + actfn);
	
#pragma acc wait

}

void Pop::propagate(float* pXs, Prj* prj) {

    /* propagate from source population activities "pXs" through projection "prj" */
    
    int Ns = prj->Ns; // can we avoid initializing scalars everytime?
    
    float tmpsum;
#pragma acc data present(pXs[:Ns], prj->pBr[:N], prj->pWrs[:N*Ns], psup[:N])
#pragma acc parallel async(1)
    {
#pragma acc loop
        for (long int r=0; r<N; r++) {		
            tmpsum = prj->pBr[r];
#pragma acc loop reduction(+:tmpsum)
            for (long int s=0; s<Ns; s++)
                tmpsum += pXs[s] * prj->pWrs[r*Ns+s];
            psup[r] = tmpsum;
        }
    }

// #pragma acc wait
    
}

void Pop::injectNoise(float nampl) {

    /* add cuda noise to support */
    
#pragma acc host_data use_device(pGnextfloat)
    
	istat = curandGenerateUniform(gt, pGnextfloat, N);
        
#pragma acc parallel loop async(1)
	for (long int r=0; r<N; r++) psup[r] += nampl * pGnextfloat[r];
    
}

void Pop::execone(float* pXs, Prj* prj, float nampl, int niter, bool monitoring) {

    /* callable function for population activity update */

    for (int iter=0; iter<niter; iter++) {

	propagate(pXs, prj);

        injectNoise(nampl);

        if (monitoring) compute_energy();
        
	updact();
	
    }

}


void Pop::compute_energy() {

    float energy = 0;
    
#pragma acc data present(psup[:N], pact[:N])
#pragma acc parallel loop reduction(+:energy)    
    for (long int n=0; n<N; n++)
        energy += - pact[n] * psup[n]; // (psup[n] - log(max(pact[n],eps)));
    
    energyhistory.push_back(energy/N);
    
}
