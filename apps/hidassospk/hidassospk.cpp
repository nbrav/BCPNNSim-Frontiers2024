/*

MIT License

Copyright (c) 2024 Anders Lansner, Naresh Ravichandran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <time.h>
#include <sys/time.h>

#include "Globals.h"
#include "Parseparam.h"
#include "Pats.h"
#include "Pop.h"
#include "Prj.h"
#include "Logger.h"

using namespace std;
using namespace Globals;

// Global parameters
long seed;
float nampl = 1e-5;
int verbosity = 1;
struct timeval start_time;

// Pop parameters
std::vector<Pop *> pops;
std::string actfn;
int Hin, Min, Hhid, Mhid, Hout, Mout, nlayer;
float maxfq, taum, again;

// Prj parameters
std::vector<Prj *> prjs;
std::vector<Prj *> ffwd_prjs, recr_prjs, fbck_prjs;
int nconnih, nconnhh, nconnhi;
int updconn_interval, updconn_nswapmax;
float updconn_threshold;
float tauz, taup;
float bgain, wgain;

// Training parameters
int nstep_gap, nstep_pat=0, nstep_ffwd, nstep_overlap, nstep_recr;
int nusupepoch, nsupepoch;

// Classifier objects
std::vector<Pop *> bcpclf_pops, lsgdclf_pops;
std::vector<Prj *> bcpclf_prjs;
std::vector<LSGD *> lsgdclf_prjs;

// Dataset parameters
int ntrpat, ntepat, nvalpat;
int binarize_in;
std::string datadir, trimgfile, teimgfile, trlblfile, telblfile, parfile;
std::string complete_teimgfile, rivalry_teimgfile, distort_teimgfile;
std::string complete_telblfile, rivalry_telblfile, distort_telblfile;
Pats *trimg = nullptr, *trlbl = nullptr, *teimg = nullptr, *telbl = nullptr;
Pats *complete_teimg, *rivalry_teimg, *distort_teimg;
Pats *complete_telbl, *rivalry_telbl, *distort_telbl;
float tracc, teacc;
float *complete_teacc, *rivalry_teacc, *distort_teacc;

// Storage parameters
bool storeacts = 1, storeweights = 1;
std::string logdir, logmodelname;
vector<Logger *> learnreg_loggers, learnlast_loggers;

float getDiffTime(struct timeval start_time)
{
    /* time difference in milli-seconds */
    struct timeval t_time;
    gettimeofday(&t_time, 0);
    return (1000.0 * (t_time.tv_sec - start_time.tv_sec) + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}

void parseparam()
{

    Parseparam *parseparam = new Parseparam(parfile);

    parseparam->postparam("seed", &seed, Long);
    parseparam->postparam("Hin", &Hin, Int);
    parseparam->postparam("Min", &Min, Int);
    parseparam->postparam("Hhid", &Hhid, Int);
    parseparam->postparam("Mhid", &Mhid, Int);
    parseparam->postparam("Hout", &Hout, Int);
    parseparam->postparam("Mout", &Mout, Int);
    parseparam->postparam("binarize_in", &binarize_in, Int);
    parseparam->postparam("Mout", &Mout, Int);
    parseparam->postparam("nconnih", &nconnih, Int);
    parseparam->postparam("nconnhh", &nconnhh, Int);
    parseparam->postparam("nconnhi", &nconnhi, Int);
    parseparam->postparam("nlayer", &nlayer, Int);
    parseparam->postparam("taum", &taum, Float);
    parseparam->postparam("tauz", &tauz, Float);
    parseparam->postparam("taup", &taup, Float);
    parseparam->postparam("maxfq", &maxfq, Float);
    parseparam->postparam("actfn", &actfn, String);
    parseparam->postparam("again", &again, Float);
    parseparam->postparam("updconn_interval", &updconn_interval, Int);
    parseparam->postparam("updconn_nswapmax", &updconn_nswapmax, Int);
    parseparam->postparam("updconn_threshold", &updconn_threshold, Float);
    parseparam->postparam("bgain", &bgain, Float);
    parseparam->postparam("wgain", &wgain, Float);
    parseparam->postparam("nstep_gap", &nstep_gap, Int);
    parseparam->postparam("nstep_ffwd", &nstep_ffwd, Int);
    parseparam->postparam("nstep_overlap", &nstep_overlap, Int);
    parseparam->postparam("nstep_recr", &nstep_recr, Int);
    parseparam->postparam("nampl", &nampl, Float);
    parseparam->postparam("nusupepoch", &nusupepoch, Int);
    parseparam->postparam("nsupepoch", &nsupepoch, Int);
    parseparam->postparam("ntrpat", &ntrpat, Int);
    parseparam->postparam("ntepat", &ntepat, Int);
    parseparam->postparam("nvalpat", &nvalpat, Int);
    parseparam->postparam("datadir", &datadir, String);
    parseparam->postparam("trimgfile", &trimgfile, String);
    parseparam->postparam("trlblfile", &trlblfile, String);
    parseparam->postparam("teimgfile", &teimgfile, String);
    parseparam->postparam("telblfile", &telblfile, String);
    parseparam->postparam("complete_teimgfile", &complete_teimgfile, String);
    parseparam->postparam("complete_telblfile", &complete_telblfile, String);
    parseparam->postparam("rivalry_teimgfile", &rivalry_teimgfile, String);
    parseparam->postparam("rivalry_telblfile", &rivalry_telblfile, String);
    parseparam->postparam("distort_teimgfile", &distort_teimgfile, String);
    parseparam->postparam("distort_telblfile", &distort_telblfile, String);
    parseparam->postparam("logdir", &logdir, String);
    parseparam->postparam("logmodelname", &logmodelname, String);

    parseparam->doparse();

    if (taup == 0)
        taup = timestep * ntrpat; // taupdt = dt / taup

    nstep_pat = nstep_ffwd + nstep_overlap + nstep_recr;

}

void initpats()
{

    trimg = new Pats(Hin, Min, datadir, trimgfile, binarize_in);
    trimg->loadpats(0, ntrpat);

    trlbl = new Pats(Hout, Mout, datadir, trlblfile);
    trlbl->loadpats(0, ntrpat);

    teimg = new Pats(Hin, Min, datadir, teimgfile, binarize_in);
    teimg->loadpats(0, ntepat);

    telbl = new Pats(Hout, Mout, datadir, telblfile);
    telbl->loadpats(0, ntepat);

    complete_teimg = new Pats(Hin, Min, datadir, complete_teimgfile, binarize_in);
    complete_teimg->loadpats(0, ntepat);

    complete_telbl = new Pats(Hout, Mout, datadir, complete_telblfile);
    complete_telbl->loadpats(0, ntepat);

    rivalry_teimg = new Pats(Hin, Min, datadir, rivalry_teimgfile, binarize_in);
    rivalry_teimg->loadpats(0, ntepat);

    rivalry_telbl = new Pats(Hout, Mout, datadir, rivalry_telblfile);
    rivalry_telbl->loadpats(0, ntepat);

    distort_teimg = new Pats(Hin, Min, datadir, distort_teimgfile, binarize_in);
    distort_teimg->loadpats(0, ntepat);

    distort_telbl = new Pats(Hout, Mout, datadir, distort_telblfile);
    distort_telbl->loadpats(0, ntepat);

    tracc = 0;
    teacc = 0;
    complete_teacc = (float *)malloc(ntepat * sizeof(float));
    rivalry_teacc = (float *)malloc(ntepat * sizeof(float));
    distort_teacc = (float *)malloc(ntepat * sizeof(float));
    for (int p = 0; p < ntepat; p++)
        complete_teacc[p] = 0;
    for (int p = 0; p < ntepat; p++)
        rivalry_teacc[p] = 0;
    for (int p = 0; p < ntepat; p++)
        distort_teacc[p] = 0;
}

void initlogs()
{

    // LEARNREGULAR LOGGERS

    for (auto prj : prjs)
    {
        if (not prj->pop_j->onthisrank())
            continue;

        int rank_i = prj->pop_i->id, rank_j = prj->pop_j->id;

        learnreg_loggers.push_back(new Logger(prj, "updconn_nswap", logdir + logmodelname + "/learn.nswap." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));
        learnreg_loggers.push_back(new Logger(prj, "conn", logdir + logmodelname + "/learn.cij." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));
        learnreg_loggers.push_back(new Logger(prj, "nmi", logdir + logmodelname + "/learn.nmi." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));

    }

    for (auto logger : learnreg_loggers)
        logger->off();

    // LEARNLAST LOGGER

    for (auto prj : prjs)
    {
        if (not prj->pop_j->onthisrank())
            continue;

        int rank_i = prj->pop_i->id, rank_j = prj->pop_j->id;

        learnlast_loggers.push_back(new Logger(prj, "pij", logdir + logmodelname + "/learn.pij." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));
        learnlast_loggers.push_back(new Logger(prj, "wij", logdir + logmodelname + "/learn.wij." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));
        learnlast_loggers.push_back(new Logger(prj, "bj", logdir + logmodelname + "/learn.bj." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));

    }

    for (auto logger : learnlast_loggers)
        logger->off();  

}

void buildnet()
{

    if (isroot() and verbosity)
    {
        printf("\nBuilding net");
        fflush(stdout);
    }

    gettimeofday(&start_time, 0);

    // adding pops
    int rank_iter = 0;
    pops.push_back(new Pop(Hin, Min, actfn, maxfq, taum, again, rank_iter));
    for (int l = 1; l < nlayer; l++)
    {
        pops.push_back(new Pop(Hhid, Mhid, actfn, maxfq, taum, again, rank_iter));
    }
    pops.push_back(new Pop(Hin, Min, actfn, maxfq, taum, again, rank_iter));

    // adding ffwd prjs
    for (int l = 1; l < nlayer; l++)
    {
        BCP *newprj = new BCP(pops[l - 1], pops[l], "BCP");
        newprj->set_tauzi(tauz);
        newprj->set_tauzj(tauz);
        newprj->set_taup(taup);
        prjs.push_back(newprj);
        ffwd_prjs.push_back(newprj);
    }

    // adding recr prjs
    for (int l = 1; l < nlayer; l++)
    {
        BCP *newprj = new BCP(pops[l], pops[l], "BCP");
        newprj->set_tauzi(tauz);
        newprj->set_tauzj(tauz);
        newprj->set_taup(taup);
        prjs.push_back(newprj);
        recr_prjs.push_back(newprj);
    }

    // adding fbck prjs
    for (int l = 1; l < nlayer; l++)
    {
        BCP *newprj = new BCP(pops[l], pops[l + 1], "BCP");
        newprj->set_tauzi(tauz);
        newprj->set_tauzj(tauz);
        newprj->set_taup(taup);
        prjs.push_back(newprj);
        fbck_prjs.push_back(newprj);
    }

    // lsgd classifiers
    for (int l = 1; l < nlayer; l++)
    {
        Pop *newpop = new Pop(Hout, Mout, "softmax", 1000, taum, 1, pops[l]->rank);
        LSGD *newprj = new LSGD(pops[l], newpop, "LSGD");
        newprj->set_tauzi(tauz);
        pops.push_back(newpop);
        prjs.push_back(newprj);
        lsgdclf_pops.push_back(newpop);
        lsgdclf_prjs.push_back(newprj);
    }

    for (auto prj : prjs)
    {
        prj->set_eps(1e-7);
        prj->set_bgain(bgain);
        prj->set_wgain(wgain);
    }

    for (auto pop : pops)
        pop->allocate_memory();
    for (auto prj : prjs)
        prj->allocate_memory();

    MPI_Barrier(MPI_COMM_WORLD);

    for (auto prj : ffwd_prjs)
    {
        if (prj->pop_i->id == 0)
            prj->initconn_rand(nconnih);
        else
            prj->initconn_rand(nconnhh);
    }
    // for (auto prj : recr_prjs)
    // {
    //     prj->initconn_rand(nconnhh);
    // }
    for (auto prj : fbck_prjs)
    {
        if (prj->pop_j->id == 0)
            prj->initconn_rand(nconnhi);
        else
            prj->initconn_rand(nconnhi);
    }

    for (auto prj : prjs)
    {
        prj->updconn_nswapmax = updconn_nswapmax;
        prj->updconn_threshold = updconn_threshold;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (verbosity and isroot())
    {
        printf("\n%-50s %10d steps %10.2f ms", "Building net done.", simstep, getDiffTime(start_time));
        fflush(stdout);
    }
}

/*
   Run one full step of network with switches for plasticity, rewiring, logging
*/
void run(bool DEBUG)
{

    for (auto pop : pops)
        pop->sync_device(); // waiting on each host for prev kernels to finish
    for (auto pop : pops)
        pop->start_send();
    for (auto pop : pops)
        pop->start_recv();
    for (auto pop : pops)
        pop->wait_and_end_send();
    for (auto pop : pops)
        pop->wait_and_end_recv();
    for (auto pop : pops)
        pop->resetsup();
    for (auto prj : prjs)
        prj->depolarize();
    for (auto pop : pops)
        pop->integrate();
    for (auto pop : pops)
        pop->inject_noise(nampl);
    for (auto pop : pops)
        pop->updact();
    for (auto prj : prjs)
        prj->updenergy();
    for (auto prj : prjs)
        prj->updtraces();
    for (auto prj : prjs)
        prj->updbw();
    for (auto prj : prjs)
        prj->updconn();
    Logger::dologall();
    advance();
}

void learn(int prn_ffwd = 0, int prn_recr = 0)
{

    if (isroot() and verbosity)
    {
        printf("\nUnsupervised learning starting..");
        fflush(stdout);
    }

    gettimeofday(&start_time, 0);

    vector<int> randids(ntrpat);
    for (int q = 0; q < ntrpat; q++)
        randids[q] = q;

    for (int epoch = 0; epoch < nusupepoch; epoch++)
    {

        // shuffle(begin(randids), end(randids), RndGen::grndgen->generator); // warning: use only on same rank

        for (int q = 0, p; q < ntrpat; q++)
        {

            p = randids[q];

            for (int t = 0; t < nstep_ffwd; t++)
            {

                float PRN = t == nstep_ffwd-1;

                // Take care of regular learn logger
                int patid = epoch * ntrpat + q;
                int pow10 = (patid>0)? pow(10, floor(log10(patid))) : 1; // find nearest power of 10, e.g. 83291 gets 10000
                int patid_on_logscale = patid % pow10 == 0;
                for (auto logger : learnreg_loggers)
                    (patid_on_logscale and PRN) ? logger->on() : logger->off();
                
                // Take care of last step learn logger
                for (auto logger : learnlast_loggers)
                    (t==nstep_ffwd-1 and q==ntrpat-1 and epoch==nusupepoch-1) ? logger->on() : logger->off();

                for (auto prj : ffwd_prjs)
                {
                    prj->bwgain = 1;
                    prj->printnow = PRN * prn_ffwd;
                    prj->REWIRE = q % updconn_interval == 0 and PRN and prn_ffwd;
                }

                for (auto prj : recr_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = PRN * prn_recr;
                    prj->REWIRE = 0; // q % updconn_interval == 0 and PRN and prn_recr; // TODO: 0
                }

                for (auto prj : fbck_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = PRN;
                    prj->REWIRE = q % updconn_interval == 0 and PRN;
                }

                for (auto prj : lsgdclf_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }

                for (auto prj : bcpclf_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }

                pops[0]->setinput(trimg->getpat(p));
                pops[2]->setinput(trimg->getpat(p));
                for (auto pop : bcpclf_pops)
                    pop->setinput(nullptr);
                for (auto pop : lsgdclf_pops)
                    pop->setinput(nullptr);

                run(true);

            }
        }

        if (isroot() and verbosity)
        {
            printf("\nEpoch %d ", epoch + 1);
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (auto logger : learnreg_loggers)
        logger->off();
    for (auto logger : learnlast_loggers)
        logger->off();

    if (verbosity and isroot())
    {
        printf("\n%-50s %10d steps %10.2f ms", "Unsupervised learning done.", simstep, getDiffTime(start_time));
        fflush(stdout);
    }
}

void learn_clfs()
{

    if (isroot() and verbosity)
    {
        printf("\nClassifier learning starting..");
        fflush(stdout);
    }

    gettimeofday(&start_time, 0);

    for (int epoch = 0; epoch < nsupepoch; epoch++)
    {

        for (int p = 0; p < ntrpat; p++)
        {

            for (int t = 0; t < nstep_ffwd; t++)
            {

                float PRN = t == nstep_ffwd - 1;

                for (auto prj : ffwd_prjs)
                {
                    prj->bwgain = 1;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }

                for (auto prj : recr_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }

                for (auto prj : fbck_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = 0;
                    prj->REWIRE = false;
                }

                for (auto prj : bcpclf_prjs)
                {
                    prj->bwgain = 0;
                    prj->printnow = PRN;
                    prj->REWIRE = false;
                }

                for (auto prj : lsgdclf_prjs)
                {
                    prj->bwgain = 1;
                    prj->printnow = PRN;
                    prj->REWIRE = false;
                }

                pops[0]->setinput(trimg->getpat(p));
                pops[2]->setinput(nullptr);
                for (auto prj : lsgdclf_prjs)
                    prj->settarget(trlbl->getpat(p));

                run(false);
            }
        }

        if (isroot() and verbosity)
        {
            printf("\nEpoch %d ", epoch + 1);
            fflush(stdout);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (verbosity and isroot())
    {
        printf("\n%-50s %10d steps %10.2f ms", "Classifier learning done.", simstep, getDiffTime(start_time));
        fflush(stdout);
    }
}

void predict(std::string dataset)
{

    if (verbosity and isroot())
    {
        printf("\n%-30s %-19s", "Prediction starting..", dataset.c_str());
        fflush(stdout);
    }

    gettimeofday(&start_time, 0);

    Pats *img, *lbl;
    int npat;
    if (dataset == "train")
    {
        img = trimg;
        lbl = trlbl;
        npat = ntrpat;
    }
    else if (dataset == "test")
    {
        img = teimg;
        lbl = telbl;
        npat = ntepat;
    }
    else if (dataset == "complete")
    {
        img = complete_teimg;
        lbl = complete_telbl;
        npat = ntepat;
    }
    else if (dataset == "rivalry")
    {
        img = rivalry_teimg;
        lbl = rivalry_telbl;
        npat = ntepat;
    }
    else if (dataset == "distort")
    {
        img = distort_teimg;
        lbl = distort_telbl;
        npat = ntepat;
    }
    else
    {
        printf("\nDataset not recognized!");
    }

    vector<Logger *> ffwd_loggers, attractor_loggers, everystep_loggers;
    if (storeacts)
    {
        for (auto prj : prjs)
        {
            if (not prj->pop_j->onthisrank())
                continue;
            int rank_i = prj->pop_i->id, rank_j = prj->pop_j->id;
            ffwd_loggers.push_back(new Logger(prj, "zi", logdir + logmodelname + "/predict.ffwd." + dataset + ".zi." + to_string(rank_i) + to_string(rank_j) + ".bin", nstep_gap + nstep_pat, nstep_gap + nstep_ffwd - 1));
            ffwd_loggers.push_back(new Logger(prj, "zj", logdir + logmodelname + "/predict.ffwd." + dataset + ".zj." + to_string(rank_i) + to_string(rank_j) + ".bin", nstep_gap + nstep_pat, nstep_gap + nstep_ffwd - 1));
            attractor_loggers.push_back(new Logger(prj, "zi", logdir + logmodelname + "/predict.attractor." + dataset + ".zi." + to_string(rank_i) + to_string(rank_j) + ".bin", nstep_gap + nstep_pat, nstep_gap + nstep_pat - 1));
            attractor_loggers.push_back(new Logger(prj, "zj", logdir + logmodelname + "/predict.attractor." + dataset + ".zj." + to_string(rank_i) + to_string(rank_j) + ".bin", nstep_gap + nstep_pat, nstep_gap + nstep_pat - 1));
        }
        for (auto pop : pops)
        {
            if (not pop->onthisrank())
                continue;
            everystep_loggers.push_back(new Logger(pop, "act", logdir + logmodelname + "/everystep." + dataset + ".act." + to_string(pop->id) + ".bin", 1));
            everystep_loggers.push_back(new Logger(pop, "sup", logdir + logmodelname + "/everystep." + dataset + ".sup." + to_string(pop->id) + ".bin", 1));
        }
        for (auto prj : prjs)
        {
            if (not prj->pop_j->onthisrank())
                continue;
            int rank_i = prj->pop_i->id, rank_j = prj->pop_j->id;
            everystep_loggers.push_back(new Logger(prj, "zi", logdir + logmodelname + "/everystep." + dataset + ".zi." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));
            everystep_loggers.push_back(new Logger(prj, "zj", logdir + logmodelname + "/everystep." + dataset + ".zj." + to_string(rank_i) + to_string(rank_j) + ".bin", 1));
        }

        for (auto logger : ffwd_loggers)
            logger->on();
        for (auto logger : attractor_loggers)
            logger->on();
    }

    for (int p = 0; p < npat; p++)
    {

        for (auto logger : everystep_loggers)
            //if (p>=5000 and p<6000)
            if (p<1000)
                logger->on();
            else
                logger->off();

        for (int t = 0; t < nstep_gap; t++)
        {

            bool FFWD_DRIVE = 1;
            bool RECR_DRIVE = 0;
            bool FBCK_DRIVE = 1;

            for (auto prj : ffwd_prjs)
            {
                prj->bwgain = FFWD_DRIVE;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : recr_prjs)
            {
                prj->bwgain = RECR_DRIVE;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : fbck_prjs)
            {
                prj->bwgain = FBCK_DRIVE;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : bcpclf_prjs)
            {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : lsgdclf_prjs)
            {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            pops[0]->setinput(nullptr);
            pops[2]->setinput(nullptr);
            for (auto pop : bcpclf_pops)
                pop->setinput(nullptr);
            for (auto pop : lsgdclf_pops)
                pop->setinput(nullptr);

            run(true);
        }

        for (int t = 0; t < nstep_pat; t++)
        {

            bool INP_DRIVE = t < nstep_ffwd + nstep_overlap;
            bool FFWD_DRIVE = t < nstep_ffwd + nstep_overlap;
            bool RECR_DRIVE = t >= nstep_ffwd; 
            bool FBCK_DRIVE = 1;

            for (auto prj : ffwd_prjs)
            {
                prj->bwgain = FFWD_DRIVE;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : recr_prjs)
            {
                prj->bwgain = RECR_DRIVE;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : fbck_prjs)
            {
                prj->bwgain = FBCK_DRIVE;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : bcpclf_prjs)
            {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            for (auto prj : lsgdclf_prjs)
            {
                prj->bwgain = 1;
                prj->printnow = 0;
                prj->REWIRE = false;
            }

            if (INP_DRIVE)
            {
                pops[0]->setinput(img->getpat(p));
            }
            else
            {
                pops[0]->setinput(nullptr);
            }
            pops[2]->setinput(nullptr);
            for (auto pop : bcpclf_pops)
                pop->setinput(nullptr);
            for (auto pop : lsgdclf_pops)
                pop->setinput(nullptr);

            run(true);
        }

        for (auto pop : lsgdclf_pops)
            if (pop->onthisrank())
            {
                if (dataset == "train")
                    tracc += pop->compute_accuracy(lbl->getpat(p));
                else if (dataset == "test")
                    teacc += 1. * pop->compute_accuracy(lbl->getpat(p));
                else if (dataset == "complete")
                    complete_teacc[p] = 1. * pop->compute_accuracy(lbl->getpat(p));
                else if (dataset == "rivalry")
                    rivalry_teacc[p] = 1. * pop->compute_accuracy(lbl->getpat(p));
                else if (dataset == "distort")
                    distort_teacc[p] = 1. * pop->compute_accuracy(lbl->getpat(p));
            }
    }

    if (storeacts)
    {
        for (auto logger : ffwd_loggers)
            logger->off();
        for (auto logger : attractor_loggers)
            logger->off();
        for (auto logger : everystep_loggers)
            logger->off();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (verbosity and isroot())
    {
        printf("\n%-30s %-19s %10d steps %10.2f ms", "Prediction done.", dataset.c_str(), simstep, getDiffTime(start_time));
        fflush(stdout);
    }
}

void summary()
{

    for (auto pop : lsgdclf_pops)
    {
        if (pop->onthisrank())
        {
            int layer = pop->id;
            printf("\nLSGD Layer %-4d Accuracy (train) = %.2f %% Accuracy (test) = %.2f %%", layer, tracc / ntrpat * 100, teacc / ntepat * 100);
            fflush(stdout);

            int ndiff = 10;
            int npat_per_diff = ntepat / ndiff;

            vector<float> complete_acc(ndiff, 0), rivalry_acc(ndiff, 0), distort_acc(ndiff, 0);
            for (int patid = 0; patid < ntepat; patid++)
            {
                int diffid = patid / npat_per_diff;
                complete_acc[diffid] += complete_teacc[patid];
                rivalry_acc[diffid] += rivalry_teacc[patid];
                distort_acc[diffid] += distort_teacc[patid];
            }
            printf("\nCOMPLETE ");
            for (int diffid = 0; diffid < ndiff; diffid++)
                printf("%.1f ", complete_acc[diffid] / 1000 * 100);
            printf("\nRIVALRY  ");
            for (int diffid = 0; diffid < ndiff; diffid++)
                printf("%.1f ", rivalry_acc[diffid] / 1000 * 100);
            printf("\nDISTORT  ");
            for (int diffid = 0; diffid < ndiff; diffid++)
                printf("%.1f ", distort_acc[diffid] / 1000 * 100);
        }
    }
}

int main(int argc, char **args)
{

    for (int i = 0; i < argc; i++)
        parfile = args[i];
    initialize_comm();
    initialize_gpu("ONE_GPU_PER_PROCESS");
    parseparam();
    gsetseed(seed);
    initpats();
    buildnet();
    initlogs();
    learn(1, 1);
    learn_clfs();
    predict("train");
    predict("test");
    predict("complete");
    predict("rivalry");
    predict("distort");
    summary();
    Logger::closeall();
    finalize_comm();
    if (isroot())
    {
        printf("\nFin.\n");
        fflush(stdout);
    }
}
