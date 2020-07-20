# CapelliniSpTRSV
A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs

## Introduction

This is the source code of a paper entitled "CapelliniSpTRSV: A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs" by Jiya Su, Feng Zhang, Weifeng Liu, Bingsheng He, Ruofan Wu, Xiaoyong Du, Rujia Wang, 2020.

The 6 algorithms in this project are execute on CUDA version. Among them, SyncFree_csc, cuSP and cuSP-layer are already proposed algorithms, which are used as benchmark algorithms. The other three algorithms are our proposed algorithms. mix has the best performance, but has a short preprocessing time. ourWrtFst has better performance than our2Part, and there is no preprocessing time for both algorithms.

We will continue to improve this project to make CapelliniSpTRSV easier to use.

## Abstract

Sparse triangular solves (SpTRSVs) have been extensively used in linear algebra fields, and many GPU SpTRSV algorithms have been proposed. Synchronization-free SpTRSVs, due to their short preprocessing time and high performance, are currently the most popular SpTRSV algorithms. However, we observe that the performance of the current synchronization-free SpTRSV algorithms on different matrices varies greatly. Specifically, we find that when the average number of components per level is high and the average number of nonzero elements per row is low, these SpTRSVs exhibit low performance. The reason is that current SpTRSVs use a warp in GPUs to process a row in sparse matrices, and such warp-level designs increase the dependencies in SpTRSVs; this problem becomes serious in these cases. To solve this problem, we propose CapelliniSpTRSV, a thread-level synchronization-free SpTRSV algorithm. Particularly, CapelliniSpTRSV has three desirable features. First, CapelliniSpTRSV does not need preprocessing to calculate levels. Second, our algorithm exhibits high performance on matrices that previous SpTRSVs cannot handle efficiently. Third, CapelliniSpTRSV is based on the most popular sparse matrix storage, compressed sparse row (CSR) format, which implies that users do not need to conduct format conversion. We evaluate CapelliniSpTRSV with 245 matrices from the Florida Sparse Matrix Collection on three GPU platforms, and experiments show that our SpTRSV exhibits 3.41 GFLOPS/s, which is 5.26x speedup over the state-of-the-art synchronization-free SpTRSV algorithm, and 4.00x speedup over the SpTRSV in cuSPARSE. CapelliniSpTRSV can be downloaded from https://github.com/JiyaSu/CapelliniSpTRSV.

## Algorithms Introduction

### SyncFree_csc

The source code is download from https://github.com/bhSPARSE/Benchmark_SpTRSM_using_CSC.

### cuSP and cuSP-layer

The code uses the SpTRSV function from the cuSPARSE library. The function has two options CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL, corresponding to cuSP and cuSP-layer respectively.

### our2Part

A novel thread-level synchronization-free SpTRSV algorithm, targeting the sparse matrices that have large number of components per level and small number of nonzero elements per row.

### ourWrtFst

An optimization SpTRSV algorithm based on our2Part, which has better performance and targets the sparse matrices that have large number of components per level and small number of nonzero elements per row.

### mix

An integrated algorithm with SyncFree and ourWrtFst for all sparse matrices.

## Execution

1. Choose the algorithm you want to run, and enter the folder corresponding to the algorithm.
1. Adjust the common.h file according to the GPU hardware, the repeated times, and the accuracy of the calculation (single or double precision).
2. Set CUDA path in the Makefile.
3. Run ``make``.
4. Run ``./main example.mtx``. (kernel is in the spts_ .h)
5. The result is saved in result.csv as ``matrix path, row number, the number of total nonzero elements, the average number of nonzero elements per row, level number, the average number of rows per level, pre_time, solve_time, gflops, bandwith``.

## Tested environments

1. nvidia GTX 1080 (Pascal) GPU in a host with CUDA v8.0 and Ubuntu 16.04.4 Linux installed.
3. nvidia Tesla V100 (Volta) GPU in a host with CUDA v9.0 and Ubuntu 16.04.1 Linux installed.
1. nvidia GeForce RTX 2080 Ti (Turing) GPU in a host with CUDA v10.2 and Ubuntu 18.04.4 Linux installed.

## Acknowledgement

CapelliniSpTRSV is developed by Renmin University of China, China University of Petroleum, National University of Singapore, and Illinois Institute of Technology.

Jiya Su, Feng Zhang, Ruofan Wu and Xiaoyong Du are with the Key Laboratory of Data Engineering and Knowledge Engineering (MOE), and School of Information, Renmin University of China.

Weifeng Liu is with the Super Scientific Software Laboratory, Department of Computer Science and Technology, China University of Petroleum.

Bingsheng He is with the School of Computing, National University of Singapore.

Rujia Wang is with the Computer Science Department, Illinois Institute of Technology.

If you have any questions, please contact us (Jiya_Su@ruc.edu.cn).

## Citation

If you use our code, please cite our paper:
```
@inproceedings{su2020sptrsv,
author = {Jiya Su and Feng Zhang and Weifeng Liu and Bingsheng He and Ruofan Wu and Xiaoyong Du and Rujia Wang},
title = {{CapelliniSpTRSV: A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs}},
year = {2020},
booktitle = {Proceedings of the 49th International Conference on Parallel Processing},
series = {ICPP '20}
}
```

