/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __Parseparam_INCLUDED__
#define __Parseparam_INCLUDED__

#include <vector>
#include <string>

// #include "Globals.h"

enum Value_t { Int = 0, Long, Float, Boole, String } ;

class Parseparam {

 public:

    Parseparam(std::string paramfile) ;

    void error(std::string errloc,std::string errstr) ;

    void postparam(std::string paramstring,void *paramvalue,Value_t paramtype) ;

    int findparam(std::string paramstring) ;

    void doparse(std::string paramlogfile = "") ;

    int getintparam(std::string paramstring,int v = 0) ;

    float getfltparam(std::string paramstring,int v = 0) ;

    long getlongparam(std::string paramstring,int v = 0) ;

    bool getbooleparam(std::string paramstring,int v = 0) ;

    std::string getstringparam(std::string paramstring,int v = 0) ;

    bool haschanged() ;

    char *timestamp() ;

    char *dolog(bool usetimestamp) ;

 protected:

    int nparampost;
    std::string _paramlogfile,_paramfile ;
    std::vector<std::string> _paramstring ;
    std::vector<void *> _paramvalue,*_vparamvalue ;
    std::vector<std::vector<int> > _vintvalue ;
    std::vector<std::vector<long> > _vlongvalue ;
    std::vector<std::vector<float> > _vfltvalue ;
    std::vector<std::vector<bool> > _vboolevalue;
    std::vector<std::vector<std::string> > _vstringvalue;
    std::vector<Value_t> _paramtype;
    time_t _oldmtime;
    char *_timestamp;

} ;

#endif // __Parseparam_INCLUDED__
