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

    b = db = m_db = v_db = m_db_corr = v_db_corr = vector<float>(Nr,0);
    w = dw = m_dw = v_dw = m_dw_corr = v_dw_corr = vector<vector<float> >(Ns,vector<float>(Nr,0));

    for (int s=0; s<Ns; s++) for (int r=0; r<Nr; r++) w[s][r] = gnextfloat() * 0.05;
    
    sup = new float[Nr];
    act = new float[Nr];
        
}

LinearSGD::~LinearSGD() {

}


void LinearSGD::storeacts(FILE* f) {

}

void LinearSGD::trainone(float* src, float* pred, float* target) {

    for (int r=0; r<Nr; r++) db[r] += target[r] - pred[r];
    
    for (int s=0; s<Ns; s++) for (int r=0; r<Nr; r++) dw[s][r] += src[s] * (target[r] - pred[r]);

    advance();
	
}

void LinearSGD::updbw() {

    if (optimizer=="Adam") {

        /* @brief update w, b from dw, db using Adam (Adaptive Moment Estimation) optimizer      
         * Original : Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980. 
         * Help : https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc  
         */

        t++;

        for (int s=0; s<Ns; s++)
            for (int r=0; r<Nr; r++) {
                
                m_dw[s][r] = beta1 * m_dw[s][r] + (1-beta1) * dw[s][r] / (float) batch_size; // Update biased first moment estimate
                v_dw[s][r] = beta2 * v_dw[s][r] + (1-beta2) * dw[s][r] * dw[s][r] / (float) batch_size ; // Update biased second raw moment estimate
                m_dw_corr[s][r] = m_dw[s][r] / (1-pow(beta1,t)) ; //  Compute bias-corrected first moment estimate
                v_dw_corr[s][r] = v_dw[s][r] / (1-pow(beta2,t)) ; // Compute bias-corrected second raw moment estimate
                w[s][r] = w[s][r] + alpha * (m_dw_corr[s][r] / (sqrt(v_dw_corr[s][r])+epsilon)) ; //  Update parameters
            }

        for (int r=0; r<Nr; r++) {

            m_db[r] = beta1 * m_db[r] + (1-beta1) * db[r] / (float) batch_size ; // Update biased first moment estimate
            v_db[r] = beta1 * v_db[r] + (1-beta2) * db[r] * db[r] / (float) batch_size ; // Update biased second raw moment estimate
            m_db_corr[r] = m_db[r] / (1-pow(beta1,t)) ; //  Compute bias-corrected first moment estimate
            v_db_corr[r] = v_db[r] / (1-pow(beta2,t)) ; // Compute bias-corrected second raw moment estimate  
            b[r] = b[r] + alpha * (m_db_corr[r] / (sqrt(v_db_corr[r])+epsilon)) ; //  Update parameters
        }

    } else {

        for (int r=0; r<Nr; r++) b[r] += alpha * db[r] / (float) batch_size;        
        for (int s=0; s<Ns; s++) for (int r=0; r<Nr; r++) w[s][r] += alpha * dw[s][r] / (float) batch_size;

    }
           

    db = vector<float>(Nr,0);    
    dw = vector<vector<float> >(Ns,vector<float>(Nr,0));
    	
}

void LinearSGD::updact() {

    for (int hr=0; hr<Hr; hr++) {

        float esupsum = 0, supmax = -1;
			
        for (int mr=0; mr<Mr; mr++) supmax = max(supmax, sup[Mr*hr+mr]);
        for (int mr=0; mr<Mr; mr++) act[Mr*hr+mr] = exp(sup[Mr*hr+mr]-supmax);
        for (int mr=0; mr<Mr; mr++) esupsum += act[Mr*hr+mr];
        for (int mr=0; mr<Mr; mr++) act[Mr*hr+mr] /= esupsum;

    }	

}

void LinearSGD::execone(float* Xs) {

    for (int r=0; r<Nr; r++) sup[r] = b[r];

    for (int s=0; s<Ns; s++) for (int r=0; r<Nr; r++) sup[r] += Xs[s] * w[s][r];

    updact();
	
}


void LinearSGD::compute_accuracy(float* pred, float* target) {

    int correct = 0;
    float predval = -FLT_MAX, targetval = -FLT_MAX;
    
    for(int r=0; r<Nr; r++) predval = max(predval, pred[r]);    
    for(int r=0; r<Nr; r++) targetval = max(targetval, target[r]);    
    for(int r=0; r<Nr; r++) correct += (target[r]==targetval) and (pred[r]==predval);
    
    ncorrect += correct;
    
}

void LinearSGD::resetncorrect() {

    ncorrect = 0;

}

float LinearSGD::getncorrect() {

  return ncorrect;

}
