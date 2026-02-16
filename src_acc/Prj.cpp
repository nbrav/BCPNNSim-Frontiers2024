/*

  Author: Anders Lansner, Naresh Ravichandran

  Created: 2021-08-02     Modified: 2021-12-06

*/

#include <vector>
#include <string>
#include <random>

#include "Globals.h"
#include "Prj.h"

#include <iostream>

using namespace std;
using namespace Globals;

int Prj::nprj = 0;

Prj::Prj(int Hs, int Ms, int Hr, int Mr, string lrule) {

    id = nprj++;

    this->Hs = Hs;
    this->Ms = Ms;
    Ns = Hs * Ms;

    this->Hr = Hr;
    this->Mr = Mr;
    Nr = Hr * Mr;

    Hrs = Hr * Hs;    
    Nrs = Nr * Ns;

    this->lrule = lrule;

    eps = 1e-7;

    lr = 0;

    Connrs = new int[Hrs];
    for (int hrs=0; hrs<Hrs; hrs++) Connrs[hrs] = 1;
    WConnrs = new int[Nrs];
    for (int rs=0; rs<Nrs; rs++) WConnrs[rs] = 1;
    mutual_info = new float[Hrs]();
    score = new float[Hrs]();

    plr = 0;

    pConnrs = new int[Hrs];
    for (int hrs=0; hrs<Hrs; hrs++) pConnrs[hrs] = 1;
    pWConnrs = new int[Nrs];
    for (int rs=0; rs<Nrs; rs++) pWConnrs[rs] = 1;
    pmutual_info = new float[Hrs]();
    pscore = new float[Hrs]();

#pragma acc enter data create(this)

#pragma acc update device(this)

#pragma acc enter data copyin(plr)
#pragma acc enter data copyin(pWConnrs[:Nrs], pConnrs[:Hrs], pscore[:Hrs], pmutual_info[:Hrs])

    initmemtraces();
    
}

Prj::~Prj() {

#pragma acc exit data delete(plr, pC, pCs[:Ns], pCr[:Nr], pCrs[:Nrs], pWrs[:Nrs], pBr[:Nr])
#pragma acc exit data delete(pWConnrs[:Nrs], pConnrs[:Hrs], pscore[:Hrs], pmutual_info[:Hrs])
    
    delete[] pCs, pCr, pCrs, pBr, pWrs;
    delete[] pConnrs, pWConnrs, pscore, pmutual_info;
    
}

void Prj::initmemtraces() {

    std::string inittype = "uniform";

    pCs = new float[Ns];
    pCr = new float[Nr];
    pCrs = new float[Nrs];
    pBr = new float[Nr];
    pWrs = new float[Nrs];
    
#pragma acc enter data copyin(pC, pCs[:Ns], pCr[:Nr], pCrs[:Nrs])
#pragma acc enter data copyin(pWrs[:Nrs], pBr[:Nr])
        
    if (inittype=="uniform") {
        
#pragma acc data present(pC, pCs[:Ns], pCr[:Nr], pCrs[:Nrs], pBr[:Nr], pWrs[:Nrs]) 
        {
#pragma acc parallel loop async(1)
            for (int s=0; s<Ns; s++) pCs[s] = eps;            
#pragma acc parallel loop async(1)
            for (int r=0; r<Nr; r++) pCr[r] = eps;            
#pragma acc parallel loop async(1)
            for (int rs=0; rs<Nrs; rs++) pCrs[rs] = eps;            
#pragma acc parallel loop async(1)
            for (int r=0; r<Nr; r++) pBr[r] = eps;            
#pragma acc parallel loop async(1)
            for (int rs=0; rs<Nrs; rs++) pWrs[rs] = eps;
        }
        
    } else if (inittype=="random") { // might too be slow on CPU, can move to GPUs

        float randfloat, norm;
        pC = 1;
        for (int s=0; s<Ns; s++) pCs[s] = 1./Ms;
        for (int r=0; r<Nr; r++) pCr[r] = 1./Mr;
        for (int r=0; r<Nr; r++)
            for (int hs=0; hs<Hs; hs++) {
                norm = 0;
                for (int ms=0; ms<Ms; ms++) {
                    randfloat = 0.25 + 0.5 * gnextfloat();
                    pCrs[r*Ns+hs*Ms+ms] = randfloat;
                    norm += randfloat;                    
                }
                for (int ms=0; ms<Ms; ms++) {
                    pCrs[r*Ns+hs*Ms+ms] = pCrs[r*Ns+hs*Ms+ms] / norm * pCr[r];
                }
            }
        for (int r=0; r<Nr; r++) pBr[r] = log(pCr[r]);
        for (int r=0; r<Nr; r++) for (int s=0; s<Ns; s++) pWrs[r*Ns+s] = log(pCrs[r*Ns+s]/(pCr[r]*pCs[s]));
                
    }

}

void Prj::store(std::string field, FILE* f) {

    if (field == "wij") {

#pragma acc update host(pWrs[:Nrs])
    fwrite(pWrs, sizeof(float), Nrs, f);

    } else if (field == "cij") {

#pragma acc update host(pCrs[:Nrs])
    fwrite(pCrs, sizeof(float), Nrs, f);

    } else if (field == "ci") {

#pragma acc update host(pCs[:Ns])
    fwrite(pCs, sizeof(float), Ns, f);

    } else if (field == "cj") {

#pragma acc update host(pCr[:Nr])
    fwrite(pCr, sizeof(float), Nr, f);

    } else if (field == "conn") {

#pragma acc update host(pConnrs[0:Hrs])
    fwrite(pConnrs, sizeof(int), Hrs, f);
    
    } else if (field == "wconn") {

#pragma acc update host(pWConnrs[:Nrs])
    fwrite(pWConnrs, sizeof(int), Nrs, f);    
    
    } else if (field == "conn") {

#pragma acc update host(pConnrs[:Hrs])
    fwrite(pConnrs, sizeof(int), Hrs, f);    
    
    } else

        printf("\nPrj::store Invalid field!");
            
}

void Prj::loadconn(std::string filename) {
    
    /* load connection matrix and set Connrs and WConnrs */

    printf("\nLoaded conn: %s", filename.c_str());
    FILE *fileptr = fopen(filename.c_str(),"rb");
    float* tmpmat = (float *)calloc(Hrs, sizeof(float));
    int nitem = fread(tmpmat, sizeof(float), Hrs, fileptr);
    
    for (int hr=0; hr<Hr; hr++)
        for (int hs=0; hs<Hs; hs++)
            pConnrs[hr*Hs + hs] = int(tmpmat[hr*Hs + hs]);    

#pragma acc update device(pConnrs[:Hrs])    

    updwconn();
    
    delete [] tmpmat;
    
}

void Prj::set_learningrate(float lr) {

  this->lr = lr;

  this->plr = lr;

#pragma acc update device(plr)  
  
}

void Prj::trainone(float* restrict pXs,float* restrict pXr) {
    
    C = (1-lr)*C + lr;

#pragma acc data present(plr, pXs[:Ns], pXr[:Nr], pCs[:Ns], pCr[:Nr], pCrs[:Nrs]) 
    {
#pragma acc parallel loop async(1)
        for (long int s=0; s<Ns; s++)
            pCs[s] = (1-plr) * pCs[s] + plr * pXs[s];

#pragma acc parallel loop async(1)
        for (long int r=0; r<Nr; r++)
            pCr[r] = (1-plr) * pCr[r] + plr * pXr[r];

#pragma acc parallel loop collapse(2) async(1)
        for (long int r=0; r<Nr; r++) 
            for (long int s=0; s<Ns; s++) {
                long int rs = r * Ns+s;
                pCrs[rs] = (1-plr) * pCrs[rs] + plr * pXs[s] * pXr[r];
            }
    }

}

void Prj::updbw() {

    if (lrule=="WILL") {
		
        // Willshaw rule
		
        for (int s=0; s<Ns; s++)
			
            for (int r=0; r<Nr; r++)
				
                Wrs[r*Ns + s] = 1<=Crs[r*Ns + s];
		
    } else if (lrule=="HOPF") {
		
        // Standard Hebb/Hopfield rule
		
        for (int s=0; s<Ns; s++)
			
            for (int r=0; r<Nr; r++)
				
                Wrs[r*Ns + s] = Crs[r*Ns + s]/C;

    } else if (lrule=="COV") {
		
        // Covariance rule
		
        for (int s=0; s<Ns; s++)
			
            for (int r=0; r<Nr; r++)
				
                Wrs[r*Ns + s] = Crs[r*Ns + s]/C - Cr[r]*Cs[s]/(C*C);
		
    } else if (lrule=="BCPNN") {
		
        // BCPNN rule
		
	float pC = C, peps = eps;
	
#pragma acc data present(pCs[:Ns], pCr[:Nr], pCrs[:Nrs], pBr[:Nr], pWrs[:Nrs], pWConnrs[:Nrs])
	{
#pragma acc parallel loop async(1)
            for (long int r=0; r<Nr; r++) {
                
                // float Pr  = max(pCr[r]/pC, peps);
                float Pr  = pCr[r]/pC + peps;
	    
                pBr[r] = log(Pr);
                
            }
            
#pragma acc parallel loop collapse(2) async(1)
            for (long int r=0; r<Nr; r++) {
                
                for (long int s=0; s<Ns; s++) {			
                    
                    long int rs = r*Ns + s;
                    
                    float Ps = max(pCs[s]/pC, peps);				
                    float Pr = max(pCr[r]/pC, peps);				
                    float Prs = max(pCrs[rs]/pC, peps*peps);
                    // float Ps = pCs[s]/pC + peps;				
                    // float Pr = pCr[r]/pC + peps;				
                    // float Prs = pCrs[rs]/pC + peps*peps;
                    
                    pWrs[rs] = log(Prs/(Ps*Pr)) * pWConnrs[rs] ;
                    
                }
                
            }
            
        }
        
    } else error("Prj::updbw","No such learning rule: " + lrule);
    
// #pragma acc wait

}

void Prj::initconn(int nconn) {

    /* initialize nconn random connections as active */

    this->nconn = nconn;

    for (int hrs=0; hrs<Hrs; hrs++) Connrs[hrs] = 0;

    for (int hr=0; hr<Hr; hr++) {
    	vector<int> hs_active = vector<int>(Hs,0);
    	for (int conn=0; conn<nconn; conn++) {	
            int hs = gnextint() % Hs;
            while (hs_active[hs]==1) hs = gnextint() % Hs;
            hs_active[hs] = 1;
    	    Connrs[hr*Hs + hs] = 1;            
    	}	
    }

    updwconn();
    
#ifdef _OPENACC

    for (int hrs=0; hrs<Hrs; hrs++) pConnrs[hrs] = Connrs[hrs];
    
#pragma acc update device(pConnrs[:Hrs])

    updwconn();
    
#endif

}

void Prj::updconn() {
	
    if (lrule != "BCPNN") error("Prj::updconn","Structural plasticity requires BCPNN learning.");

    // calculate mutual info score
	
    float pC = C, peps = eps;	

#pragma acc data present(pCs[:Ns], pCr[:Nr], pCrs[:Nrs])
#pragma acc parallel loop collapse(2)
    for (long int hr = 0; hr < Hr; hr++) {
        for (long int hs = 0; hs < Hs; hs++) {
            
            float tmpsum = 0;
#pragma acc loop collapse(2) reduction(+:tmpsum)	    
            for (long int mr = 0; mr < Mr; mr++) {
        	for (long int ms = 0; ms < Ms; ms++) {
		    
        	    long int r = hr * Mr + mr;
        	    long int s = hs * Ms + ms;
        	    long int rs = r * Ns + s;        
        	    float Ps = max(pCs[s]/pC, peps);
        	    float Pr = max(pCr[r]/pC, peps);			
        	    float Prs = max(pCrs[rs]/pC, peps*peps);
                    // float Ps = pCs[s]/pC + peps;
                    // float Pr = pCr[r]/pC + peps;			
                    // float Prs = pCrs[rs]/pC + peps*peps;
        	    tmpsum += Prs * log(Prs/(Ps*Pr));
		    
        	}        
            }
	    
            pmutual_info[hr*Hs+hs] = tmpsum;
            
        }
    }
    
    // iterate sequentially over recv. hypercolumn, compute score and do flips
           
    for (long int hr=0; hr<Hr; hr++) {
        
        // (re)compute score from mutual info

#pragma acc data present(pConnrs[:Hrs], pWConnrs[:Nrs], pmutual_info[:Hrs])
#pragma acc parallel loop        
        for (long int hs=0; hs<Hs; hs++) {
            
            int fanout = 0;
#pragma acc loop reduction(+:fanout)
            for (long int hr2=0; hr2<Hr; hr2++)
                fanout += pConnrs[hr2*Hs+hs]==1;
#pragma acc loop 
            for (long int hr2=0; hr2<Hr; hr2++)
                pscore[hr2*Hs+hs] = pmutual_info[hr2*Hs+hs] / (fanout + 1);		
        }
        
        // update connections
        
        bool converged = false;
               
        for (int swapid=0; swapid<updconn_nswap and not converged; swapid++) { 
            
            float active_minscore = FLT_MAX, silent_maxscore = -FLT_MAX;            
            long int active_id, silent_id; 

            // find active hypercolumn with minimum score
#pragma acc data present(pConnrs[:Hrs], pscore[:Hrs])
#pragma acc parallel loop reduction(min:active_minscore)   
            for (long int hs=0; hs<Hs; hs++) {                
                if (pConnrs[hr*Hs+hs]==1 and pscore[hr*Hs+hs] < active_minscore) {
                    active_minscore = pscore[hr*Hs+hs];
                    active_id = hs;
                }
            }

            // find silent hypercolumn with maximum score
#pragma acc data present(pConnrs[:Hrs], pscore[:Hrs])
#pragma acc parallel loop reduction(max:silent_maxscore)
            for (long int hs=0; hs<Hs; hs++) {                
                if (pConnrs[hr*Hs+hs]==0 and pscore[hr*Hs+hs] >= silent_maxscore) {
                    silent_maxscore = pscore[hr*Hs+hs];
                    silent_id = hs;
                }
            }

            // check for convergence
            converged = silent_maxscore <= updconn_threshold * active_minscore;

            if (converged) break;

            // flip a connection pair
#pragma acc data present(pConnrs[:Hrs])
#pragma acc parallel loop
// #pragma acc loop seq
            for (long int hs=0; hs<Hs; hs++) {
                
                if (hs==active_id) pConnrs[hr*Hs+hs] = 0;
                if (hs==silent_id) pConnrs[hr*Hs+hs] = 1;
                
            }
        }
    }    

    // update wconn from conn
    updwconn();

}

void Prj::updwconn() {
        
    /* Comput WConnrs from Connrs */
    
#pragma acc data present(pConnrs[:Hrs], pWConnrs[:Nrs])
#pragma acc parallel loop collapse(4)
    for (long int hr=0; hr<Hr; hr++)
        for (long int hs=0; hs<Hs; hs++)            
            for (long int mr=0; mr<Mr; mr++)
                for (long int ms=0; ms<Ms; ms++) {
                    
                    long int r = hr * Mr + mr;
                    long int s = hs * Ms + ms;
                    
                    pWConnrs[r*Ns+s] = (pConnrs[hr*Hs+hs]==1)*1;
                    
                }
    
}
