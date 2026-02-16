/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#include <vector>
#include <string>
#include <random>

#include "Globals.h"
#include "LinearSGD.h"

#include <iostream>

using namespace std;
using namespace Globals;

LinearSGD::LinearSGD(int Hs,int Ms,int Hr,int Mr,string optimizer) {

    this->Hs = Hs;
    this->Ms = Ms;
    Ns = Hs * Ms;
    this->Hr = Hr;
    this->Mr = Mr;
    Nr = Hr * Mr;
    
    this->optimizer = optimizer;

    sup = new float[Nr]();
    act = new float[Nr]();
    b = new float[Nr]();
    db = new float[Nr]();
    m_db = new float[Nr]();
    v_db = new float[Nr]();
    m_db_corr = new float[Nr]();
    v_db_corr = new float[Nr]();
    w = new float[Ns*Nr]();
    dw = new float[Ns*Nr]();
    m_dw = new float[Ns*Nr]();
    v_dw = new float[Ns*Nr]();
    m_dw_corr = new float[Ns*Nr]();
    v_dw_corr = new float[Ns*Nr]();
    ncorrect = 0;
    
    psup = new float[Nr]();
    pact = new float[Nr]();
    pb = new float[Nr]();
    pdb = new float[Nr]();
    pm_db = new float[Nr]();
    pv_db = new float[Nr]();
    pm_db_corr = new float[Nr]();
    pv_db_corr = new float[Nr]();
    pw = new float[Ns*Nr]();
    pdw = new float[Ns*Nr]();
    pm_dw = new float[Ns*Nr]();
    pv_dw = new float[Ns*Nr]();
    pm_dw_corr = new float[Ns*Nr]();
    pv_dw_corr = new float[Ns*Nr]();
    pncorrect = 0;
        
    for (int rs=0; rs<Nr*Ns; rs++) w[rs] = gnextfloat() * 0.05;
    
#pragma acc enter data create(this)
    
#pragma acc update device(this)

#pragma acc enter data copyin(pncorrect)
#pragma acc enter data copyin(pb[0:Nr], pdb[0:Nr], pw[0:Ns*Nr], pdw[0:Ns*Nr])
#pragma acc enter data copyin(pv_db[0:Nr], pv_db_corr[0:Nr], pv_dw[0:Ns*Nr], pv_dw_corr[0:Ns*Nr])
#pragma acc enter data copyin(pm_db[0:Nr], pm_db_corr[0:Nr], pm_dw[0:Ns*Nr], pm_dw_corr[0:Ns*Nr])
#pragma acc enter data copyin(psup[0:Nr], pact[0:Nr])

}

LinearSGD::~LinearSGD() {

#pragma acc exit data delete(pncorrect)
#pragma acc exit data delete(pb[0:Nr], pdb[0:Nr], pw[0:Ns*Nr], pdw[0:Ns*Nr])
#pragma acc exit data delete(pm_db[0:Nr], pv_db[0:Nr], pm_db_corr[0:Nr], pv_db_corr[0:Nr])
#pragma acc exit data delete(pm_dw[0:Ns*Nr], pv_dw[0:Ns*Nr], pm_dw_corr[0:Ns*Nr], pv_dw_corr[0:Ns*Nr])
#pragma acc exit data delete(psup[0:Nr], pact[0:Nr])

    delete[] pb, pdb, pw, pdw;
    delete[] pm_db, pv_db, pm_db_corr, pv_db_corr;
    delete[] pm_dw, pv_dw, pm_dw_corr, pv_dw_corr;
    delete[] psup, pact;

}

float* LinearSGD::getact() {

    return pact;

}

void LinearSGD::storeacts(FILE* f) {

#pragma acc update host(pact[0:Nr]) 

    fwrite(pact, sizeof(float), Nr, f);
}

void LinearSGD::trainone(float* restrict psrc, float* restrict ppred, float* restrict ptarget) {

#pragma acc data present(psrc[0:Ns], ppred[0:Nr], ptarget[0:Nr], pdb[0:Nr], pdw[0:Nr*Ns]) 
    {
#pragma acc parallel loop async(1)
        for (int r=0; r<Nr; r++) pdb[r] += ptarget[r] - ppred[r];

#pragma acc parallel loop collapse(2) async(1)
        for (int r=0; r<Nr; r++) 
            for (int s=0; s<Ns; s++) {
                int rs = r * Ns+s;
                pdw[rs] += psrc[s] * (ptarget[r] - ppred[r]) ;
            }
    }

}

void LinearSGD::updbw() {

    t++;

    float pbeta1_t = (1-pow(beta1,t)), pbeta2_t = (1-pow(beta2,t));

#pragma acc data present(pdb[0:Nr], pdw[0:Nr*Ns], pb[0:Nr], pw[0:Nr*Ns])
    {
#pragma acc parallel loop collapse(2) async(1)
        for (int r=0; r<Nr; r++)
            for (int s=0; s<Ns; s++) {
	
                int rs = r*Ns + s;
	
                pm_dw[rs] = beta1 * pm_dw[rs] + (1-beta1) * pdw[rs] / (float) batch_size; // Update biased first moment estimate
                pv_dw[rs] = beta2 * pv_dw[rs] + (1-beta2) * pdw[rs] * pdw[rs] / (float) batch_size ; // Update biased second raw moment estimate
                pm_dw_corr[rs] = pm_dw[rs] / pbeta1_t ; // Compute bias-corrected first moment estimate
                pv_dw_corr[rs] = pv_dw[rs] / pbeta2_t ; // Compute bias-corrected second raw moment estimate
                pw[rs] = pw[rs] + alpha * (pm_dw_corr[rs] / (sqrt(pv_dw_corr[rs])+epsilon)) ; //  Update parameters
                pdw[rs] = 0; // Reset gradients            	    
	
            }
    
#pragma acc parallel loop async(1)
        for (int r=0; r<Nr; r++) {
      
            pm_db[r] = beta1 * pm_db[r] + (1-beta1) * pdb[r] / (float) batch_size ; // Update biased first moment estimate
            pv_db[r] = beta1 * pv_db[r] + (1-beta2) * pdb[r] * pdb[r] / (float) batch_size ; // Update biased second raw moment estimate
            pm_db_corr[r] = pm_db[r] / pbeta1_t ; // Compute bias-corrected first moment estimate
            pv_db_corr[r] = pv_db[r] / pbeta2_t ; // Compute bias-corrected second raw moment estimate  
            pb[r] = pb[r] + alpha * (pm_db_corr[r] / (sqrt(pv_db_corr[r])+epsilon)) ; //  Update parameters
            pdb[r] = 0; // Resets gradients
      
        }
    }

}

void LinearSGD::updact() {

#pragma acc parallel loop async(1)
    for (int hr=0; hr<Hr; hr++) {

        float  supmax = psup[Mr*hr],esupsum = 0;
#pragma acc loop reduction(max:supmax)	    
        for (int mr=0; mr<Mr; mr++)	supmax = max(supmax,psup[Mr*hr+mr]);
#pragma acc loop 	    
        for (int mr=0; mr<Mr; mr++) pact[Mr*hr+mr] = exp(psup[Mr*hr+mr]-supmax);
#pragma acc loop reduction(+:esupsum)	    
        for (int mr=0; mr<Mr; mr++) esupsum += pact[Mr*hr+mr];
#pragma acc loop 	    
        for (int mr=0; mr<Mr; mr++) pact[Mr*hr+mr] /= esupsum;

    }       

}

void LinearSGD::execone(float* restrict pXs) {

    float tmp = 0;

#pragma acc data present(pXs[0:Ns], pb[0:Nr], pw[0:Nr*Ns])
    {
#pragma acc parallel loop async(1)
	for (int r=0; r<Nr; r++) {
	    tmp = pb[r];
#pragma acc loop reduction(+:tmp)
	    for (int s=0; s<Ns; s++) 
                tmp += pXs[s] * pw[r*Ns+s];
	    psup[r] = tmp;
	}
    }
    
    updact();

}


void LinearSGD::compute_accuracy(float* restrict ptarget) {

    float* restrict ppred = this->pact;

    float predval = -FLT_MAX, targetval = -FLT_MAX;
    
#pragma acc data present (ppred[0:Nr], ptarget[0:Nr], pncorrect)
    {
#pragma acc parallel loop reduction(max:predval) async(1)
        for(int r=0; r<Nr; r++) predval = max(predval, ppred[r]);
#pragma acc parallel loop reduction(max:targetval) async(1)
        for(int r=0; r<Nr; r++) targetval = max(targetval, ptarget[r]);
        int correct = 0;
#pragma acc parallel loop reduction(+:correct) async(1)
        for(int r=0; r<Nr; r++) correct += (ptarget[r]==targetval) and (ppred[r]==predval);
#pragma acc wait(1)
        ncorrect += correct;
    }
    
}

void LinearSGD::resetncorrect() {

    ncorrect = 0;
    pncorrect = 0;
  
#pragma acc update device(pncorrect)
  
}

float LinearSGD::getncorrect() {

    return ncorrect;

}
