/*

  Author: Anders Lansner, Naresh Balaji

  Created: 2021-08-10     Modified: 2021-08-10

*/

#ifndef __LinearSGD_included
#define __LinearSGD_included

#include <vector>
#include <string>
#include <random>
#include <cfloat>

#include "Globals.h"


class LinearSGD {

public:

    int Hs, Ms, Ns, Hr, Mr, Nr;
    int batch_size=100, ncorrect=0;
    float t=0, beta1=0.9, beta2=0.999, epsilon=1e-7f, alpha=0.001;
    std::string optimizer;

    float *act, *sup;
    float *b, *db, *w, *dw;
    float *m_db, *v_db, *m_db_corr, *v_db_corr, *m_dw, *v_dw, *m_dw_corr, *v_dw_corr;

    float* restrict pact;
    float* restrict psup;    
    float* restrict pb;
    float* restrict pdb;  
    float* restrict pw;
    float* restrict pdw;  
    float* restrict pm_db;
    float* restrict pv_db;
    float* restrict pm_db_corr;
    float* restrict pv_db_corr;
    float* restrict pm_dw;
    float* restrict pv_dw;
    float* restrict pm_dw_corr;
    float* restrict pv_dw_corr;
    float pncorrect; 

    LinearSGD(int Hs,int Ms,int Hr,int Mr,std::string optimizer = "Adam");
    ~LinearSGD();
    float* getact();
    void storeacts(FILE*); 
    void trainone(float*, float*, float*);
    void updbw();
    void updact();
    void execone(float*);
    void compute_accuracy(float*);
    void resetncorrect();
    float getncorrect();

} ;

#endif // __LinearSGD_included
