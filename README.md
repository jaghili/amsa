# A-MSA
An implementation of the depth-to-deep adaptive algorithm in Python

# Requirements
the code requires `mpi4py` (with MPI routines), `numpy` and `cmaes` (optional).

# Usage
The sinus test case is preconfigured for the sinus test case and can be launched using

```
mpirun --host localhost:16 --np 16 python main.py
```

where 16 is the number of available CPU threads.
