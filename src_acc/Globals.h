/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#ifndef __Globals_included
#define __Globals_included

#include <vector>
#include <string>
#include <random>

namespace Globals {

    extern int simstep;

    extern float timestep,simtime;

    void error(std::string errloc,std::string errstr,int errcode = -1);

    void warning(std::string warnloc,std::string warnstr);

    void reset();

    void advance();

    void gsetseed(long seed) ;

    int gnextint() ;

    float gnextfloat() ;

    int argmax(std::vector<float> vec,int i1,int n);

    int argmin(std::vector<float> vec,int i1,int n);

    float vlen(std::vector<float> vec) ;

    float vdiff(std::vector<float> vec1,std::vector<float> vec2) ;

    float vl1(std::vector<float> vec1,std::vector<float> vec2) ;

    void tofile(std::vector<float> vec,FILE *outfp);

    void tofile(std::vector<float> vec,std::string filename);

    void tofile(std::vector<std::vector<float> > mat,FILE *outfp);

    void tofile(std::vector<std::vector<float> > mat,std::string filename) ;

    void tofile(std::vector<int> vec,FILE *outfp);

    void tofile(std::vector<int> vec,std::string filename);

    void tofile(std::vector<std::vector<int> > mat,FILE *outfp);

    void tofile(std::vector<std::vector<int> > mat,std::string filename) ;

} ;

class RndGen {

protected:

    long seed ;
    std::mt19937_64 generator ;

    std::uniform_real_distribution<float> uniformfloatdistr ;
    std::uniform_int_distribution<int> uniformintdistr ;
    std::poisson_distribution<int> poissondistr ;
    
public:

    static RndGen *grndgen;

    RndGen(long seedoffs = -1) ;

    void setseed(long seed,int hcuid = -1) ;

    void setpoissonmean(float mean) ;

    long getseed() ;

    int nextint() ;

    float nextfloat() ;

    int nextpoisson() ;

} ;

#endif // __Globals_included
