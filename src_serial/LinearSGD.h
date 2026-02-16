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

    std::string optimizer;

    std::vector<float> b, db;

    std::vector<std::vector<float> > w, dw;

    float *sup, *act;

    std::vector<float> m_db, v_db, m_db_corr, v_db_corr;
    
    std::vector<std::vector<float> > m_dw, v_dw, m_dw_corr, v_dw_corr;

    float t=0, beta1=0.9, beta2=0.999, epsilon=1e-7f, alpha=0.001;
    // float t=0, beta1=0.9, beta2=0.999, epsilon=1e-7f, alpha=2e-5f;

    int batch_size=100, ncorrect=0; // Accuracy train h1o = 79.18 h2o = 95.71 Accuracy test h1o = 71.91 h2o = 79.89
    // int batch_size=20, ncorrect=0; // Accuracy train h1o = 95.59 h2o = 99.04 Accuracy test h1o = 77.77 h2o = 80.00

    LinearSGD(int Hs,int Ms,int Hr,int Mr,std::string optimizer = "Adam");

     ~LinearSGD();

    void storeacts(FILE*);
  
    void trainone(float*, float*, float*);

    void updbw();

    void updact();

    void execone(float*);

    void compute_accuracy(float*, float*);

    void resetncorrect();

    float getncorrect();

    float* getact() { return act; }

private:

} ;

#endif // __LinearSGD_included
