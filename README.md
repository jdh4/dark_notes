# Dark Matter Notes

## Creating the Conda Environment

```bash
$ ssh <YourNetID>@tigergpu.princeton.edu  # or adroit
$ mkdir -p software/dark  # or another location
$ cd software/dark
$ wget https://raw.githubusercontent.com/jdh4/dark_notes/master/dark_env.sh
$ bash dark_env.sh | tee build.log
```

You can then do:

```
$ module load anaconda3/2020.2
$ conda activate dark-env
$ python -c "import torch; print(torch.__version__)"
1.5.0
$ python Pylians3/Tests/import_libraries.py  # no output means success
```

The following warning can be ignored:

```
--------------------------------------------------------------------------
WARNING: There are more than one active ports on host 'tigergpu', but the
default subnet GID prefix was detected on more than one of these
ports.  If these ports are connected to different physical IB
networks, this configuration will fail in Open MPI.  This version of
Open MPI requires that every physically separate IB subnet that is
used between connected MPI processes must have different subnet ID
values.

Please see this FAQ entry for more details:

  http://www.open-mpi.org/faq/?category=openfabrics#ofa-default-subnet-gid

NOTE: You can turn off this warning by setting the MCA parameter
      btl_openib_warn_default_gid_prefix to 0.
--------------------------------------------------------------------------
```

## Adding NVIDIA Apex to the Environment (Optional)

The [Apex](https://github.com/nvidia/apex) library allows for automatic mixed-precision training and distributed training:

```
$ ssh <YourNetID>@adroit.princeton.edu  # mixed precision only possible on V100
$ module load anaconda3/2020.2 rh/devtoolset/8 cudatoolkit/10.1
$ conda activate dark-env
$ cd software/dark
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ export TORCH_CUDA_ARCH_LIST="7.0"
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Submitting a Job to TigerGPU

Create a Slurm scipt such as this (job.slurm):

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.2
conda activate dark-env

python _run_graph_net_nv.py
```

To submit the job:

```
$ sbatch job.slurm
```

Monitor the state of the job (queued, running, finished) with:

```
$ squeue -u $USER
```

Consider adding this alias to your `.bashrc` file:

```
alias sq='squeue -u $USER'
```
## Submitting a Job to Adroit

Create a Slurm scipt such as this (job.slurm):

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                # total memory per node
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=hackathon  # REMOVE THIS AFTER THE HACKATHON

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.2
conda activate dark-env

python _run_graph_net_nv.py
```

## Interactive Allocations

For a 30-minutes interactive allocation with 4 CPU-cores, 8 GB of CPU memory and 1 GPU:

```
$ salloc -N 1 -n 4 -t 30 --mem=8G --gres=gpu:1 
```

Or equivalently:

```
$ salloc --nodes=1 --ntasks=4 --time 30:00 --mem=8G --gres=gpu:1
```

## Nsight Systems for Profiling

Nsight Systems [getting started guide](https://docs.nvidia.com/nsight-systems/) and notes on [Summit](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#profiling-gpu-code-with-nvidia-developer-tools).

IMPORTANT: Do not run profiling jobs in your /home directory because it has a 20 GB quota. Instead launch jobs from `/scratch/gpfs` where you have 500 GB. Here's an example:

```
$ ssh <YourNetID>@tigergpu.princeton.edu
$ cd /scratch/gpfs/<YourNetID>
$ mkdir myjob && cd myjob
# prepare Slurm script
$ sbatch job.slurm
```

Below is an example Slurm script:

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.2
conda activate dark-env

nsys profile -f true --stats=true python _run_graph_net_nv.py
```

You can either download the `.qdrep` file to your local machine to use `nsight-sys` to view the data or do `ssh -X tigressdata.princeton.edu` and use `nsight-sys` on that machine. The latter approach would look like this:

```
# in a new terminal
$ ssh -X <YourNetID>@tigressdata.princeton.edu
$ cd /tiger/scratch/gpfs/<YourNetID>/myjob
$ nsight-sys report1.qdrep
```

## Large Cache

Nsight Systems will cache lots of data. If you encouter file quota problems then you may need to clear this directory:

```
~/.nsightsystems
```

## Nsight Compute for GPU Kernel Profiling (Adroit or Traverse but not TigerGPU)

See the NVIDIA [documentation](https://docs.nvidia.com/nsight-compute/). This tool does not support the P100 GPUs of TigerGPU.

```
module load cudatoolkit/10.2
nv-nsight-cu-cli  # or nv-nsight-cu for GUI
```

Below is a sample slurm script:

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.2
conda activate dark-env

/usr/local/cuda-10.2/bin/nv-nsight-cu-cli -f -o my_report_${SLURM_JOBID} python _run_graph_net_nv.py
```

One can then use `nv-nsight-cu` to view the results:

```
# ssh -X adroit
$ module load cudatoolkit/10.2
$ nv-nsight-cu my_report_xxxxxx.nsight-cuprof-report
```

## line_prof for Profiling

The [line_prof](https://github.com/rkern/line_profiler) tool provides profiling info for each line of a function.

First add the `@profile` decorator to the function(s) in the Python script:

```python
import numpy as np

@profile
def minimum_distance():
  dist_min = 1e300
  for i in range(N - 1):
    for j in range(i + 1, N):
      dist = abs(x[i] - x[j])
      if (dist < dist_min): dist_min = dist
  return dist_min

@profile
def mygeo():
  return np.cos(np.sin(x))

N = 3000
x = np.random.randn(N)

y = mygeo()
z = minimum_distance()
```

Submit the job (sbatch job.slurm):

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=30G                # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.2
conda activate dark-env

kernprof -l _run_graph_net_nv.py
```

Examine the results:

```
# module load anaconda3/2020.2
# conda activate dark-env
$ python -m line_profiler _run_graph_net_nv.py.lprof

Timer unit: 1e-06 s

Total time: 6.60265 s
File: _run_graph_net_nv.py
Function: minimum_distance at line 3

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     3                                           @profile
     4                                           def minimum_distance():
     5         1          1.0      1.0      0.0    dist_min = 1e300
     6      3000       1002.0      0.3      0.0    for i in range(N - 1):
     7   4501499    1478080.0      0.3     22.4      for j in range(i + 1, N):
     8   4498500    3565623.0      0.8     54.0        dist = abs(x[i] - x[j])
     9   4498500    1557944.0      0.3     23.6        if (dist < dist_min): dist_min = dist
    10         1          1.0      1.0      0.0    return dist_min

Total time: 0.000161 s
File: myscript.py
Function: mygeo at line 12

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    12                                           @profile
    13                                           def mygeo():
    14         1        161.0    161.0    100.0    return np.cos(np.sin(x))
```

## PyTorch at Princeton

[https://researchcomputing.princeton.edu/pytorch](https://researchcomputing.princeton.edu/pytorch)

## Monitoring GPU Utilization

See the procedure at the bottom of [this page](https://researchcomputing.princeton.edu/tigergpu-utilization). Consider using this alias which will put you on the compute node where your most recent job is running:

```
goto() { ssh $(squeue -u $USER | tail -1 | tr -s [:blank:] | cut -d' ' --fields=9); }
```

From there you can run `nvidia-smi` or `gpustat`.

## Intro to GPU Programming at Princeton

[https://github.com/PrincetonUniversity/gpu_programming_intro](https://github.com/PrincetonUniversity/gpu_programming_intro)

## DDT and MAP

We have the DDT parallel debugger which can be used for CUDA kernels. MAP is a general purpose profiler which can provide info on CUDA kernels and MPI.

## Convert a Jupyter Notebook to a Python Script

```
$ jupyter nbconvert --to script run_graph_net_nv.ipynb
```

## Use a Command-Line Python Debugger with Syntax Highlighting

```
$ ipython -m ipdb _run_graph_net_nv.py
```

## Links

[https://github.com/anonml2020/gn](https://github.com/anonml2020/gn)   
[https://github.com/franciscovillaescusa/Pylians3](https://github.com/franciscovillaescusa/Pylians3)   
