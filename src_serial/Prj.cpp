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

    initmemtraces();
    
}

Prj::~Prj() {

}

void Prj::initmemtraces() {

    std::string inittype = "uniform";

    Cs = new float[Ns];
    Cr = new float[Nr];
    Crs = new float[Nrs];
    Br = new float[Nr];
    Wrs = new float[Nrs];
    
    if (inittype=="uniform") {
        
        C = eps;    
        for (int r=0; r<Nr; r++) Cr[r] = eps;
        for (int s=0; s<Ns; s++) Cs[s] = eps;
        for (int rs=0; rs<Nrs; rs++) Crs[rs] = eps;
        for (int r=0; r<Nr; r++) Br[r] = eps;
        for (int rs=0; rs<Nrs; rs++) Wrs[rs] = eps;
        
    } else if (inittype=="random") { // might too be slow on CPU, can move to GPUs

        float randfloat, norm;
        C = 1;
        for (int s=0; s<Ns; s++) Cs[s] = 1./Ms;
        for (int r=0; r<Nr; r++) Cr[r] = 1./Mr;
        for (int r=0; r<Nr; r++)
            for (int hs=0; hs<Hs; hs++) {
                norm = 0;
                for (int ms=0; ms<Ms; ms++) {
                    randfloat = 0.25 + 0.5 * gnextfloat();
                    Crs[r*Ns+hs*Ms+ms] = randfloat;
                    norm += randfloat;                    
                }
                for (int ms=0; ms<Ms; ms++) {
                    Crs[r*Ns+hs*Ms+ms] = Crs[r*Ns+hs*Ms+ms] / norm * Cr[r];
                }
            }
        for (int r=0; r<Nr; r++) Br[r] = log(Cr[r]);
        for (int r=0; r<Nr; r++) for (int s=0; s<Ns; s++) Wrs[r*Ns+s] = log(Crs[r*Ns+s]/(Cr[r]*Cs[s]));
                
    }

}

void Prj::store(std::string field, FILE* f) {

    if (field == "wij") {

        fwrite(Wrs, sizeof(float), Nrs, f);    

    } else if (field == "cij") {

        fwrite(Crs, sizeof(float), Nrs, f);

    } else if (field == "ci") {

    fwrite(Cs, sizeof(float), Ns, f);

    } else if (field == "cj") {

        fwrite(Cr, sizeof(float), Nr, f);

    } else if (field == "conn") {

        fwrite(Connrs, sizeof(int), Hrs, f);    
    
    } else if (field == "wconn") {

        fwrite(WConnrs, sizeof(int), Nrs, f);    
    
    } else if (field == "conn") {

        fwrite(Connrs, sizeof(int), Hrs, f);    
    
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
            Connrs[hr*Hs + hs] = int(tmpmat[hr*Hs + hs]); 
    
    delete [] tmpmat;
    
}

void Prj::set_learningrate(float lr) {

  this->lr = lr;

}

void Prj::trainone(float* Xs, float* Xr) {
    
    C = (1-lr)*C + lr;

    for (int s=0; s<Ns; s++) Cs[s] = (1-lr)*Cs[s] + lr*Xs[s];

    for (int r=0; r<Nr; r++) Cr[r] = (1-lr)*Cr[r] + lr*Xr[r];

    for (int r=0; r<Nr; r++)

	for (int s=0; s<Ns; s++)
		
	    Crs[r*Ns + s] = (1-lr)*Crs[r*Ns + s] + lr*Xs[s]*Xr[r];

    advance();
	
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
		
        float Ps,Pr,Prs;
		
        for (int r=0; r<Nr; r++) {

            Pr  = max(Cr[r]/C,eps);
			
            Br[r] = log(Pr);
			
        }
			
        for (int s=0; s<Ns; s++) {
			
            for (int r=0; r<Nr; r++) {

                Ps = max(Cs[s]/C, eps);
				
                Pr = max(Cr[r]/C, eps);
				
                Prs = max(Crs[r*Ns + s]/C, eps*eps);
				
                Wrs[r*Ns + s] = log(Prs/(Pr*Ps)) * WConnrs[r*Ns + s];
				
            }
			
        }
		
    } else error("Prj::updbw","No such learning rule: " + lrule);
    
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
    
}


void Prj::updconn() {
	
    if (lrule != "BCPNN") error("Prj::updconn","Structural plasticity requires BCPNN learning.");

    // calculate mutual info score
	
    float Ps,Pr,Prs;	

    for (int r=0; r<Nr; r++) {
            
	for (int s=0; s<Ns; s++) {
        
            Ps = max(Cs[s]/C, eps);
            Pr = max(Cr[r]/C, eps);			
            Prs = max(Crs[r*Ns+s]/C, eps*eps);
            
            mutual_info[r/Mr * Hs + s/Ms]+= Prs * log(Prs/(Ps*Pr)) ;
            
        }        
    }
    
    for (int hr=0; hr<Hr; hr++) {

        // (re)compute score from mutual info
        
        for (int hs=0; hs<Hs; hs++) {
            
            int fanout = 0;
            for (int hr2=0; hr2<Hr; hr2++) fanout += Connrs[hr2*Hs + hs]==1;
            for (int hr2=0; hr2<Hr; hr2++) score[hr2*Hs + hs] = mutual_info[hr2*Hs + hs] / (fanout + 1);
            
        }
        
        // update connections
        
        bool converged = false;
               
        for (int swapid=0; swapid<updconn_nswap and not converged; swapid++) { 
            
            int active_id, silent_id;
            
            float active_minscore = FLT_MAX, silent_maxscore = -FLT_MAX;
            
            for (int hs=0; hs<Hs; hs++) {
                
                if (Connrs[hr*Hs+hs]==1 and score[hr*Hs+hs] < active_minscore)
		    { active_id = hs ; active_minscore = score[hr*Hs+hs]; }

                if (Connrs[hr*Hs+hs]==0 and score[hr*Hs+hs] > silent_maxscore)
		    { silent_id = hs ; silent_maxscore = score[hr*Hs+hs]; }

            }

            if (silent_maxscore > updconn_threshold * active_minscore) {

                Connrs[hr*Hs + active_id] = 0;                
                Connrs[hr*Hs + silent_id] = 1;
                
            } else converged = true;
            
        }	
    }    

    updwconn();
    
}

void Prj::updwconn() {

    /* Compute WConnrs from Connrs  */

    for (int hs=0; hs<Hs; hs++)
	for (int hr=0; hr<Hr; hr++)            
	    for (int r=hr*Mr; r<(hr+1)*Mr; r++)
	        for (int s=hs*Ms; s<(hs+1)*Ms; s++)
                    WConnrs[r*Ns + s] = Connrs[hr*Hs+hs]==1;
    
}
