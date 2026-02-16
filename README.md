# BCPNNSim

Bayesian Confidence Propagation Neural Network Simulator

This repository provides code for the experiments in the paper 

Ravichandran, N., Lansner, A., & Herman, P. (2024). Spiking representation learning for associative memories. Frontiers in Neuroscience, 18, 1439414. (https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439414/full)

The code is implemented in C++, with MPI for message passing and CUDA for GPU parallelization. It can also be converted easilily to HIP and run on AMD GPUs (https://rocm.docs.amd.com/projects/HIP/en/develop/user_guide/hip_porting_guide.html).

# Extract MNIST dataset
```
cd Data/mnist/
python3 extract.py
cd ../../
```

# Compile and Run
```
make -f Makefile.cuda.ws
./apps/hidassospk/hidassospk ./apps/hidassospk/hidassospk.par
```
The code was developed and tested on CUDA workstation equipped with NVIDIA RTX A4000 GPUs. The code was compiled with cuda/11.5, cuBLAS, hipBlas, MPICH. 

HIP files are also provided as makefile and source files for reference, but have not been thoroughly tested.

# References

Ravichandran, N., Lansner, A., & Herman, P. (2024). Spiking representation learning for associative memories. Frontiers in Neuroscience, 18, 1439414.