# CapelliniSpTRSV
A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs
<br><hr>
<h3>Introduction</h3>

This is the source code of a paper entitled "CapelliniSpTRSV: A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs" by Jiya Su, Feng Zhang, Weifeng Liu, Bingsheng He, Ruofan Wu, Xiaoyong Du, Rujia Wang, 2020.

The 6 algorithms in this project are execute on CUDA version. Among them, 2_SyncFree_csc2, 3_cuSP2 and 4_cuSP-layer2 are already proposed algorithms, which are used as benchmark algorithms. The other three algorithms are our proposed algorithms. 8_mix2 has the best performance, but has a short preprocessing time. 7_ourWrtFst2 has better performance than 6_our2Part2, and there is no preprocessing time for both algorithms.

<br><hr>
<h3>Algorithms Introduction</h3>

- 2_SyncFree_csc2

The source code is download from https://github.com/bhSPARSE/Benchmark_SpTRSM_using_CSC.

- 3_cuSP2 and 4_cuSP-layer2

The code uses the SpTRSV function from the cuSPARSE library. The function has two options CUSPARSE_SOLVE_POLICY_NO_LEVEL and CUSPARSE_SOLVE_POLICY_USE_LEVEL, corresponding to 3_cuSP2 and 4_cuSP-layer2 respectively.

- 6_our2Part2

A novel thread-level synchronization-free SpTRSV algorithm, targeting the sparse matrices that have large number of components per level and small number of nonzero elements per row.

- 7_ourWrtFst2

An optimization SpTRSV algorithm based on 6_our2Part2, which has better performance and targets the sparse matrices that have large number of components per level and small number of nonzero elements per row.

- 8_mix2

An integrated algorithm with SyncFree and ourWrtFst for all sparse matrices.

<h3>Execution</h3>

1. Adjust the common.h file according to the GPU hardware and the accuracy of the calculation (single or double precision),
2. Set CUDA path in the Makefile,
3. Run ``make``,
4. Run ``./main example.mtx``. 
(kernel is in the spts_ .h)

<h3>Tested environments</h3>

1. nvidia GTX 1080 (Pascal) GPU in a host with CUDA v8.0 and Ubuntu 16.04.4 Linux installed.
3. nvidia Tesla V100 (Volta) GPU in a host with CUDA v9.0 and Ubuntu 16.04.1 Linux installed.
1. nvidia GeForce RTX 2080 Ti (Turing) GPU in a host with CUDA v10.2 and Ubuntu 18.04.4 Linux installed.



