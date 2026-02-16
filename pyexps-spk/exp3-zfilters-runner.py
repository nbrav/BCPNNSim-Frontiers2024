#!/usr/bin/env python
from utils import *

def run(args, params, paramstr) :

    # File structure: resultsdir/paramstr/dt_str/

    # Find time stamp
    timestamp = time.time()
    jobtag = str(int(timestamp)%1000)
    datatimedir = datetime.now().strftime("%Y-%m-%d_%H:%M:%S:%f")
    
    # Create resultsdir
    os.system(f"mkdir -p {args.resultsdir}")
    
    # Create resultsdir/paramstr
    os.system(f"mkdir -p {args.resultsdir}/{paramstr}")

    # Create resultsdir/datatimedir
    newdir = f"{args.resultsdir}/{paramstr}/{datatimedir}"
    os.system("mkdir -p " + newdir)

    # modify par file with custom values
    newpar = []
    with open(parentdir + args.parfile, "r") as ifile:
        for line in ifile:
            splitline = line.split()
            if len(splitline) >= 2 and splitline[0] in params.keys():
                newpar.append( line.replace(splitline[1], params[splitline[0]]) )
            else:
                newpar.append( line )
    ifile.close()
    
    with open(newdir + "/net.par", "w") as ofile:
        ofile.write("".join(newpar))
    ofile.close()

    # make sbatch file
    context = {
        "jobname": args.jobname + jobtag,
        "maxdur": args.maxdur,
        "directory": newdir,
        "exefile": "net.exe",
        "parfile": "net.par"
    } 

    mkbatchscript(newdir, context)

    os.chdir(newdir)
    os.system("ln -s " + parentdir + "/Data ")
    os.system(f"cp {parentdir}/{args.exefile} net.exe")
    os.system("pwd")
    if args.hpc=="vega" or args.hpc=="dardel":
        os.system("sbatch run.sba")
    elif args.hpc == "ws":
        os.system(f"time mpirun -n 2 net.exe net.par")
    else:
        print ("Unknown HPC!")
    os.chdir(parentdir)

def mkbatchscript(directory, context) :

    if args.hpc == "dardel":
        template = \
"""#!/bin/bash -l
#SBATCH --job-name={jobname}
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --account=naiss2023-5-484 
#SBATCH --time={maxdur}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
export MPICH_GPU_SUPPORT_ENABLED=1
ml rocm/5.0.2 craype-accel-amd-gfx90a
CURRENT_DIR={directory}
cd $CURRENT_DIR
time srun {exefile} {parfile}
"""
    elif args.hpc == "vega":
        template = \
"""#!/bin/bash -l
#SBATCH --job-name={jobname}
#SBATCH --output=test.out
#SBATCH --error=test.out
#SBATCH --account=r2207-203-users
#SBATCH --time={maxdur}
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=40GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
CURRENT_DIR={directory}
cd $CURRENT_DIR
time mpirun {exefile} {parfile}
"""
        
    with open(directory + '/run.sba','w') as file:
        file.write(template.format(**context))
    file.close()
    
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation", "-a", help = "Allocation id", default = "naiss2023-5-484")
    parser.add_argument("--jobname", "-j", help = "Jobname string", default = "icortex")
    parser.add_argument("--maxdur", "-t", help = "Max duration string (HH:NN:SS)", default = "10:00:00")
    parser.add_argument("--exefile", "-e", help = "Executable file", default = "apps/hidassospk/hidassospk")
    parser.add_argument("--parfile", "-p", help = "Parameter file path", default = "apps/hidassospk/hidassospk.par")
    parser.add_argument("--resultsdir", "-d", help = "Resultsdir path", default = "/cfs/klemming/scratch/n/nbrav/logs/")
    parser.add_argument("--ntasks", "-n", help = "Ntasks argument",type = int, default = 1)
    parser.add_argument("--comment", "-c", help = "Comment string", default = "none")
    parser.add_argument("--hpc", help = "hpc system", default = "dardel")
    parser.add_argument("--verbosity", "-v", help = "Verbosity level", default = 0, action = "count")
    args = parser.parse_args()
    
    if args.hpc == "ws":
        parentdir = "/home/naresh/Desktop/HiddenAssociativeMemoryGPU/"
    elif args.hpc == "dardel":
        parentdir = "/cfs/klemming/home/n/nbrav/Private/HiddenAssociativeMemoryGPU/"
    elif args.hpc == "vega":
        parentdir = "/ceph/hpc/home/eunareshr/HiddenAssociativeMemoryGPU/"
        
    paramtemplate = "exp3-zfilter-logs-mnist60k"
    allparam = {
        "taum": [str(x) for x in [0.005]],
        "tauz": [str(x) for x in [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]],
        "taup": [str(x) for x in [5]],
        "maxfq": [str(x) for x in [20, 50, 100, 200, 500, 1000]],
        "actfn": ['stcind'],
        "nstep_gap": [str(x) for x in [100]],
        "nstep_ffwd": [str(x) for x in [100]],
        "nstep_overlap": [str(x) for x in [50]],
        "nstep_recr": [str(x) for x in [150]],
        "nusupepoch": [str(20)],
        "nsupepoch": [str(0)],
        "ntrpat": [str(6000)],
        "ntepat": [str(10000)],
        "logdir": ["./"],
        "logmodelname": ["."],
        "seed": [str(x) for x in range(123451, 123452)],
    }
    nrun = 0
    for param in [dict(zip(allparam, v)) for v in product(*allparam.values())]:
        nrun += 1
        paramstr = paramtemplate.format(**param)
        run(args, param, paramstr)
    print ("\nStarted nrun=%4d"%nrun)

