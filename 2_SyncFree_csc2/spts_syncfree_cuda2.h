#ifndef _SPTS_SYNCFREE_CUDA2_
#define _SPTS_SYNCFREE_CUDA2_

#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>


__global__
void sptrsv_syncfree_cuda_analyser(const int   *d_cscRowIdx,
                                   const int    m,
                                   const int    nnz,
                                   int   *d_graphInDegree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&d_graphInDegree[d_cscRowIdx[global_id]], 1);
    }
}


__global__
void sptrsv_syncfree_cuda_executor_update(const int*         d_cscColPtr,
                                          const int*         d_cscRowIdx,
                                          const VALUE_TYPE*  d_cscVal,
                                          int*                           d_graphInDegree,
                                          VALUE_TYPE*                    d_left_sum,
                                          const int                      m,
                                          const int                      substitution,
                                          const VALUE_TYPE*  d_b,
                                          VALUE_TYPE*                    d_x,
                                          int*                           d_while_profiler,
                                          int*                           d_id_extractor)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int global_x_id = 0;
    if (!lane_id)
        global_x_id = atomicAdd(d_id_extractor, 1);
    global_x_id = __shfl(global_x_id, 0);
    
    if (global_x_id >= m) return;
    
    // substitution is forward or backward
    global_x_id = substitution == SUBSTITUTION_FORWARD ?
    global_x_id : m - 1 - global_x_id;
    
    // Prefetch
    const int pos = substitution == SUBSTITUTION_FORWARD ?
    d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    //asm("prefetch.global.L2 [%0];"::"d"(d_cscVal[d_cscColPtr[global_x_id] + 1 + lane_id]));
    //asm("prefetch.global.L2 [%0];"::"r"(d_cscRowIdx[d_cscColPtr[global_x_id] + 1 + lane_id]));
    
    // Consumer
    do {
        __threadfence_block();
    }
    while (d_graphInDegree[global_x_id] != 1);
    
    VALUE_TYPE xi = d_left_sum[global_x_id];
    xi = (d_b[global_x_id] - xi) * coef;
    
    // Producer
    const int start_ptr = substitution == SUBSTITUTION_FORWARD ?
    d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ?
    d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];
        
        atomicAdd(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        __threadfence();
        atomicSub(&d_graphInDegree[rowIdx], 1);
    }
    
    //finish
    if (!lane_id) d_x[global_x_id] = xi;
}


int sptrsv_syncfree_cuda(const int           *cscColPtrTR,
                         const int           *cscRowIdxTR,
                         const VALUE_TYPE    *cscValTR,
                         const int            m,
                         const int            n,
                         const int            nnzTR,
                         VALUE_TYPE    *x,
                         const VALUE_TYPE    *b,
                         const VALUE_TYPE    *x_ref,
                         double              *pre_time_add,
                         double              *solve_time_add,
                         double              *gflops_add,
                         double              *bandwidth_add
                         )
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }
    
    // transfer host mem to device mem
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    VALUE_TYPE *d_cscValTR;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;
    
    int rhs=1;
    
    // Matrix L
    cudaMalloc((void **)&d_cscColPtrTR, (n+1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR  * sizeof(int));
    cudaMalloc((void **)&d_cscValTR,    nnzTR  * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, (n+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, nnzTR  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR,    cscValTR,    nnzTR  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);
    
    // Vector b
    cudaMalloc((void **)&d_b, m  * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * rhs * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    
    // Vector x
    cudaMalloc((void **)&d_x, n * rhs * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * rhs * sizeof(VALUE_TYPE));
    
    //  - cuda syncfree SpTRSV analysis start!
    //printf(" - cuda syncfree SpTRSV analysis start!\n");
    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    
    // malloc tmp memory to generate in-degree
    int *d_graphInDegree;
    int *d_graphInDegree_backup;
    cudaMalloc((void **)&d_graphInDegree, m * sizeof(int));
    cudaMalloc((void **)&d_graphInDegree_backup, m * sizeof(int));
    
    int *d_id_extractor;
    cudaMalloc((void **)&d_id_extractor, sizeof(int));
    
    int num_threads = 128;
    int num_blocks = ceil ((double)nnzTR / (double)num_threads);
    
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        cudaMemset(d_graphInDegree, 0, m * sizeof(int));
        sptrsv_syncfree_cuda_analyser<<< num_blocks, num_threads >>>
        (d_cscRowIdxTR, m, nnzTR, d_graphInDegree);
    }
    cudaDeviceSynchronize();
    
    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;
    
    *pre_time_add=time_cuda_analysis;
    
    //printf("cuda syncfree SpTRSV analysis on L used %4.2f ms\n", time_cuda_analysis);
    
    //  - cuda syncfree SpTRSV solve start!
    //printf(" - cuda syncfree SpTRSV solve start!\n");
    
    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *d_left_sum;
    cudaMalloc((void **)&d_left_sum, sizeof(VALUE_TYPE) * m * rhs);
    
    // backup in-degree array, only used for benchmarking multiple runs
    cudaMemcpy(d_graphInDegree_backup, d_graphInDegree, m * sizeof(int), cudaMemcpyDeviceToDevice);
    
    // this is for profiling while loop only
    int *d_while_profiler;
    cudaMalloc((void **)&d_while_profiler, sizeof(int) * n);
    cudaMemset(d_while_profiler, 0, sizeof(int) * n);
    int *while_profiler = (int *)malloc(sizeof(int) * n);
    
    // step 5: solve L*y = x
    double time_cuda_solve = 0;
    int  substitution=SUBSTITUTION_FORWARD;
    
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // get a unmodified in-degree array, only for benchmarking use
        cudaMemcpy(d_graphInDegree, d_graphInDegree_backup, m * sizeof(int), cudaMemcpyDeviceToDevice);
        //cudaMemset(d_graphInDegree, 0, sizeof(int) * m);
        
        // clear left_sum array, only for benchmarking use
        cudaMemset(d_left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);
        cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * n * rhs);
        cudaMemset(d_id_extractor, 0, sizeof(int));
        
        gettimeofday(&t1, NULL);
        
        
            num_threads = WARP_PER_BLOCK * WARP_SIZE;
            //num_threads = 1 * WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            //sptrsv_syncfree_cuda_executor<<< num_blocks, num_threads >>>
            sptrsv_syncfree_cuda_executor_update<<< num_blocks, num_threads >>>
            (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
             d_graphInDegree, d_left_sum,
             m, substitution, d_b, d_x, d_while_profiler, d_id_extractor);
        
//        else
//        {
//            num_threads = 4 * WARP_SIZE;
//            num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
//            sptrsm_syncfree_cuda_executor_update<<< num_blocks, num_threads >>>
//            (d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
//             d_graphInDegree, d_left_sum,
//             m, substitution, rhs, opt,
//             d_b, d_x, d_while_profiler, d_id_extractor);
//        }
        
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        
        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    
    time_cuda_solve /= BENCH_REPEAT;
    double flop = 2*(double)rhs*(double)nnzTR;
    double dataSize = (double)((n+1)*sizeof(int) + (nnzTR+n)*sizeof(int) + nnzTR*sizeof(VALUE_TYPE) + 2*n*sizeof(VALUE_TYPE));
    
    *solve_time_add=time_cuda_solve;
    *gflops_add=flop/(1e6*time_cuda_solve);
    *bandwidth_add=dataSize/(1e6*time_cuda_solve);
    
//    printf("cuda syncfree SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
//           time_cuda_solve, flop/(1e6*time_cuda_solve));
    //*gflops = flop/(1e6*time_cuda_solve);
    
    cudaMemcpy(x, d_x, n * rhs * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    
    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;
    
    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
        //if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;
    
    if (res < accuracy)
        //printf("cuda syncfree SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
        ;
    else
    {
        printf("cuda syncfree SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);
        *solve_time_add=-1;
    }
    
    // profile while loop
    cudaMemcpy(while_profiler, d_while_profiler, n * sizeof(int), cudaMemcpyDeviceToHost);
    long long unsigned int while_count = 0;
    for (int i = 0; i < n; i++)
    {
        while_count += while_profiler[i];
        //printf("while_profiler[%i] = %i\n", i, while_profiler[i]);
    }
    //printf("\nwhile_count= %llu in total, %llu per row/column\n", while_count, while_count/m);
    
    // step 6: free resources
    free(while_profiler);
    
    cudaFree(d_graphInDegree);
    cudaFree(d_graphInDegree_backup);
    cudaFree(d_id_extractor);
    cudaFree(d_left_sum);
    cudaFree(d_while_profiler);
    
    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);
    
    return 0;
}





#endif

