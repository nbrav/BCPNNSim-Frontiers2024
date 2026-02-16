/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#include "Globals.h"
#include "Pats.h"

using namespace std;
using namespace Globals;


Pats::Pats(int H, int M, std::string dir, std::string filename, bool binarize, std::string patype) {

    this->H = H;
    this->M = M;
    this->N = H * M;
    this->dir = dir;
    this->filename = filename;
    this->binarize = binarize;
    this->patype = patype;

}

void Pats::mkbinpats(int npat) {

    /* make binary patterns, either orthogonal or random */

    this->pbegin = 0;
    this->pend = npat;
    this->npat = npat;

    pats = (float *)calloc(npat * N, sizeof(float));

    for (int p=0; p<npat; p++) 

        if (patype=="ortho") {

            for (int h=0; h<H; h++) pats[p*N + h*M + p%M] = 1;

        } else if (patype=="rand") {

            for (int h=0; h<H; h++) pats[p*N + h*M + gnextint()%M] = 1;

        } else error("Pats::mkbinpats","No such patype: " + patype);

#ifdef _OPENACC
#pragma acc enter data copyin(pats[0:npat*N])
#endif                
  
}

void Pats::distortpats(string distype,int disarg) {

    /* distort patterns with noise */
    
    if (disarg>H) error("Pats::distopats","Illegal disarg>H: H = " + to_string(H) + " disarg = " +
         		to_string(disarg));

    dispats = (float *)calloc(npat * N, sizeof(float));

    for (int p=0; p<npat; p++) 
        for (int i=0; i<N; i++)
            dispats[p*N + i] = pats[p*N + i];

    float* tmp = (float *)calloc(H, sizeof(float));
    int h;

    for (size_t p=0; p<npat; p++) {

        for (int d=0; d<H; d++) tmp[d] = 0;
        
        if (distype=="hblank") {
            
            for (int d=0; d<disarg; d++) {				
                h = gnextint()%H;				
                while (tmp[h]==1) h = gnextint()%H;		
                for (int m=0; m<M; m++) dispats[p*N + h*M + m] = 0;				
                tmp[h] = 1;				
            }
			
        } else if (distype=="nflip") {
			
            for (int d=0; d<disarg; d++) {				
                h = gnextint()%H;
                while (tmp[h]==1) h = gnextint()%H;				
                for (int m=0; m<M; m++) dispats[p*N + h*M + m] = 0;				
                dispats[p*N + h*M + gnextint()%M] = 1;				
                tmp[h] = 1;				
            }
			
        } else error("Pats::distopats","No such distype: " + distype);
		
    }

#ifdef _OPENACC
#pragma acc enter data copyin(dispats[0:npat*N])
#endif                
  
}

void Pats::loadpats(int qbegin, int qend) {

    /* load binary patterns in range of query begin to end */

    if (this->pbegin == qbegin and this->pend == qend) return; // we already got this data; do nothing

    clearpats();

    pbegin = qbegin;
    pend = qend;
    npat = pend - pbegin;

    pats = (float *)calloc(npat * N, sizeof(float));
    FILE *fileptr = fopen((dir+filename).c_str(),"rb");

    if (binarize) {

        if (M!=2) fprintf(stderr, "Warning! Layer not binary HCUs but binarizing data.");
        
        fseek(fileptr, pbegin * H * sizeof(float), SEEK_SET);
        float* tmppats = (float *)calloc(npat * H, sizeof(float));
        int nitem = fread(tmppats, sizeof(float), npat * H, fileptr);

        for (int p=0; p<npat; p++)
            for (int h=0; h<H; h++) {
                pats[p*N + h*M + 0] = tmppats[p*H + h];
                pats[p*N + h*M + 1] = 1 - tmppats[p*H + h];
            }

        delete [] tmppats;

        // printf("\nLoaded %10s [begin=%d, end=%d, npat=%d] N=%d nitem=%d ", filename.c_str(), pbegin, pend, npat, N, nitem/N);

    } else {

        fseek(fileptr, pbegin * N * sizeof(float), SEEK_SET);
        int nitem = fread(pats, sizeof(float), npat * N, fileptr);

        // printf("\nLoaded %10s [begin=%d, end=%d, npat=%d] N=%d nitem=%d ", filename.c_str(), pbegin, pend, npat, N, nitem/N);
        
    }
    
    fclose(fileptr);

#ifdef _OPENACC
#pragma acc enter data copyin(pats[0:npat*N])
#endif                
  
}

void Pats::clearpats() {

    if (npat<=0) return;
        
    delete [] pats;
        
#ifdef _OPENACC
#pragma acc exit data delete(pats[0:npat*N])
#endif

}

float* Pats::getpat(int p) {

    if (pbegin>p or p>=pend) { fprintf(stderr, "Warning! Data %d not loaded within range [%d, %d]!", p, pbegin, pend); }

    return &(pats[(p-pbegin) * N]);
    
}
