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

}

Pop::~Pop() {
    
}

void Pop::store(std::string field, FILE* f) {

    if (field == "act") {
        
        fwrite(act, sizeof(float), N, f);    
    
    } else if (field == "sup") {

        fwrite(sup, sizeof(float), N, f);    

    } else
        
        printf("\nPop::store Invalid field!");
            
}

void Pop::updact() {

    if (actfn=="WTA")

        error("Pop::updact","actfn=='WTA' not yet implemented");
    
    else if (actfn=="kWTA")
		
        error("Pop::updact","actfn=='kWTA' not yet implemented");
	
    else if (actfn=="softWTA") {

        for (int h=0; h<H; h++) {

            float supmax = -FLT_MAX, esupsum = 0;
			
            for (int m=0; m<M; m++) supmax = max(supmax, sup[M*h+m]);
            for (int m=0; m<M; m++) act[M*h+m] = exp(sup[M*h+m]-supmax);
            for (int m=0; m<M; m++) esupsum += act[M*h+m];
            for (int m=0; m<M; m++) act[M*h+m] /= esupsum;

        }
			  	  
    } else if (actfn=="HALF")
		
        error("Pop::updact","actfn=='HALF' not yet implemented");
	
    else
		
        error("Pop::updact","No such actfn: " + actfn);
	
}

void Pop::propagate(float* Xs, Prj* prj) {

    /* propagate from source population activities "Xs" through projection "prj" */

    int Ns = prj->Ns;
    
    for (int r=0; r<N; r++) {
	
        sup[r] = prj->Br[r];
      
    	for (int s=0; s<Ns; s++)
	    
            sup[r] += Xs[s] * prj->Wrs[r*Ns + s];

    }
    
}

void Pop::injectNoise(float nampl) {

    /* add noise to support */
    
    for (int r=0; r<N; r++) sup[r] += nampl * gnextfloat();
    
}

void Pop::execone(float* Xs, Prj* prj, float nampl, int niter, bool monitoring) {

    /* callable function for population activity update */

    for (int iter=0; iter<niter; iter++) {

	propagate(Xs, prj);

        injectNoise(nampl);

	updact();
	
    }

}
