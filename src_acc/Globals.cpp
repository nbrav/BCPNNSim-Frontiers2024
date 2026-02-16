/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#include <stdio.h>

#include "Globals.h"

using namespace std;
using namespace Globals;

int Globals::simstep = 0;

float Globals::timestep = 0.001,Globals::simtime = simstep * timestep;

void Globals::error(string errloc,string errstr,int errcode) {

    fprintf(stderr,"ERROR in %s: %s\n",errloc.c_str(),errstr.c_str());

    exit(errcode);

}


void Globals::warning(string warnloc,string warnstr) {

    fprintf(stderr,"WARNING in %s: %s\n",warnloc.c_str(),warnstr.c_str());

}

  
void Globals::reset() {

    simstep = 0;

    simtime = 0;

}

void Globals::gsetseed(long seed) { RndGen::grndgen->setseed(seed); }


int Globals::gnextint()  { return RndGen::grndgen->nextint(); }


float Globals::gnextfloat() { return RndGen::grndgen->nextfloat(); }


void Globals::advance() {

    simstep += 1;

    simtime = simstep * timestep;

}


int Globals::argmax(vector<float> vec,int i1,int n) {

    int maxi = i1;
    float maxv = vec[maxi];

    for (int i=i1+1; i<i1+n; i++) if (vec[i]>maxv) { maxi = i; maxv = vec[i]; }

    return maxi;

}


int Globals::argmin(vector<float> vec,int i1,int n) {

    int mini = i1;
    float minv = vec[mini];

    for (int i=i1+1; i<i1+n; i++) if (vec[i]<minv) { mini = i; minv = vec[i]; }

    return mini;

}


float Globals::vlen(vector<float> vec) {

    float vlen = 0;

    for (size_t i=0; i<vec.size(); i++) vlen += vec[i]*vec[i];

    return sqrt(vlen);

}


float Globals::vdiff(vector<float> vec1,vector<float> vec2) {

    if (vec1.size()!=vec2.size()) error("Globals::vdiff","vec1 -- vec2 length mismatch");

    float vdiff = 0;

    for (size_t i=0; i<vec1.size(); i++) vdiff += (vec2[i]-vec1[i])*(vec2[i]-vec1[i]);

    return sqrt(vdiff);

}


float Globals::vl1(vector<float> vec1,vector<float> vec2) {

    if (vec1.size()!=vec2.size()) error("Globals::vl1","vec1 -- vec2 length mismatch");

    float vl1 = 0;

    for (size_t i=0; i<vec1.size(); i++) vl1 += abs(vec2[i]-vec1[i]);

    return vl1;

}


void Globals::tofile(vector<float> vec,FILE *outfp) {

    fwrite (vec.data(),sizeof(float),vec.size(),outfp);    

}


void Globals::tofile(vector<float> vec,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(vec,outfp);

    fclose(outfp);

}


void Globals::tofile(vector<vector<float> > mat,FILE *outfp) {

    for (size_t r=0; r<mat.size(); r++) tofile(mat[r],outfp);

}


void Globals::tofile(vector<vector<float> > mat,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(mat,outfp);

    fclose(outfp);

}

void Globals::tofile(vector<int> vec,FILE *outfp) {

    fwrite (vec.data(),sizeof(int),vec.size(),outfp);    

}


void Globals::tofile(vector<int> vec,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(vec,outfp);

    fclose(outfp);

}

void Globals::tofile(vector<vector<int> > mat,FILE *outfp) {

    for (size_t r=0; r<mat.size(); r++) tofile(mat[r],outfp);

}

void Globals::tofile(vector<vector<int> > mat,string filename) {

    FILE *outfp = fopen(filename.c_str(),"wb");

    tofile(mat,outfp);

    fclose(outfp);

}

RndGen *RndGen::grndgen = new RndGen();

RndGen::RndGen(long seedoffs) {

    uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);
 
    uniformintdistr = uniform_int_distribution<int>();
 
    setseed(4711174,seedoffs);

}


void RndGen::setseed(long seed,int hcuid) {

    // seed==0 gives random seed
    if (seed==0) seed = random_device{}();
    this->seed = seed + 17 * (hcuid + 1);
    generator.seed(this->seed);

}

long RndGen::getseed() { return seed; }

int RndGen::nextint() { return uniformintdistr(generator); }

float RndGen::nextfloat() { return uniformfloatdistr(generator); }
