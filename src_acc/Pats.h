/*

  Author: Anders Lansner

  Created: 2021-08-02     Modified: 2021-08-02

*/

#ifndef __Pats_included
#define __Pats_included

#include <vector>
#include <string>
#include <random>

class Pats {

 public:

    std::string patype, dir, filename;
    int H, M, N;
    int pbegin=0, pend=-1, npat=-1;
    bool binarize;
    float *pats, *dispats;

 public:

    Pats(int H, int M, std::string dir = "", std::string filename = "", bool binarize = false, std::string patype = "rand");

    // std::vector<std::vector<float> > getpats();

    void mkbinpats(int npat);

    void distortpats(std::string distype = "nflip",int disarg = 1);

    void loadpats(int, int);

    void clearpats();

    float* getpat(int);

} ;

#endif // __Pats_included
