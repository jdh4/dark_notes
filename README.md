## Creating the Environment

```bash
$ ssh <YourNetID>@tigergpu.princeton.edu
$ mkdir -p software/dark  # or another name
$ cd software/dark
$ cp /scratch/gpfs/jdh4/dark_env.sh .
$ bash dark_env.sh | tee build.log
```

## Submitting a Job

Create a Slurm scipt such as this (job.slurm):

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2020.2
conda activate dark-env

python myscript.py
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

## Interactive allocations

For a 30-minutes interactive allocation with 4 CPU-cores, 8 GB of CPU memory and 1 GPU:

```
$ salloc -N 1 -n 4 -t 30 --mem=8G --gres=gpu:1 
```

Or equivalently:

```
$ salloc --nodes=1 --ntasks=4 --time 30:00 --mem=8G --gres=gpu:1
```

## Nsight Systems

```
#!/bin/bash
#SBATCH --job-name=dark          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2020.2
conda activate dark-env

nsys profile -f true --stats=true python myscript.py
```

You can either download the profiling file to your local machine to use `nsight-sys` to view the data or do `ssh -X tigressdata.princeton.edu` and use nsight-sys on that machine.

## Tips on Using PyTorch at Princeton

[https://researchcomputing.princeton.edu/pytorch](https://researchcomputing.princeton.edu/pytorch)

## DDT and MAP

We have the DDT parallel debugger which can be used for CUDA kernels. MAP is a general purpose profiler which can provide info on CUDA kernels and MPI calls.
