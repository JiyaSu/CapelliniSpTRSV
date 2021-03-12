#ifndef _SPTS_CUSPARSE_
#define _SPTS_CUSPARSE_

#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cusparse.h>
#include <assert.h>

    

int spts_cuSPARSE(const int           *csrRowPtrL_tmp,
                       const int           *csrColIdxL_tmp,
                       const VALUE_TYPE    *csrValL_tmp,
                       const int            m,
                       const int            n,
                       const int            nnzL,
                  double                    *pre_time_add,
                  double                    *solve_time_add,
                  double                    *gflops_add,
                  double                    *bandwidth_add
                  )
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }
    
    VALUE_TYPE *x_ref;
    VALUE_TYPE *b ;
    
    get_x_b(m, n, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, &x_ref, &b);
    //printf("%g\n",csrValL_tmp[0]);
    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    
    
    
    cusparseHandle_t handle = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    
    //Create handle
    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    

    // transfer host mem to device mem
    int *d_csrRowPtrL;
    int *d_csrColIdx;
    VALUE_TYPE *d_csrValL;
    VALUE_TYPE *d_b;
    VALUE_TYPE *d_x;
    
    // Matrix L
    cudaMalloc((void **)&d_csrRowPtrL, (m+1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdx, nnzL  * sizeof(int));
    cudaMalloc((void **)&d_csrValL,    nnzL  * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_csrRowPtrL, csrRowPtrL_tmp, (m+1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, csrColIdxL_tmp, nnzL  * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValL,    csrValL_tmp,    nnzL  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice);
    
    // Vector b
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_b, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    
    // Vector x
    cudaMalloc((void **)&d_x, n  * sizeof(VALUE_TYPE));
    cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE));
    
    //describe the shape and properties of a matrix
    // step 1: create a descriptor which contains
    // - matrix L indices is zero
    // - matrix L is lower triangular
    // - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
    //   (L may not have all diagonal elements.)
    cusparseMatDescr_t descr = 0;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);
    
    
    // step 2: create a empty info structure
    csrsv2Info_t info = 0;
    cusparseCreateCsrsv2Info(&info);
    
    // step 3: query how much memory used in csrsv2, and allocate the buffer
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int pBufferSize;
    cusparseDcsrsv2_bufferSize(handle, trans, m, nnzL, descr,
                               d_csrValL, d_csrRowPtrL, d_csrColIdx, info, &pBufferSize);
    
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    void *pBuffer = 0;
    cudaMalloc((void**)&pBuffer, pBufferSize);
    
    
    //printf(" - cuda syncfree SpTS analysis start!\n");
    struct timeval t1, t2;
    int i;
    
    // step 4: perform analysis
    
//    gettimeofday(&t1, NULL);
//    for (i = 0; i < BENCH_REPEAT; i++)
//    {
//        cusparseScsrsv2_analysis(handle, trans, m, nnzL, descr,
//                                 d_csrValL, d_csrRowPtrL, d_csrColIdx,
//                                 info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
//        cudaDeviceSynchronize();
//
//    }
//    gettimeofday(&t2, NULL);
//    double time_cuda_analysis = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
//    time_cuda_analysis /= BENCH_REPEAT;
//
//    *pre_time_add=time_cuda_analysis;
    //printf("cudaSPARSE SpTS analysis without layer on L used %4.2f ms\n", time_cuda_analysis);
    
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    gettimeofday(&t1, NULL);
    for (i = 0; i < BENCH_REPEAT; i++)
    {
        cusparseDcsrsv2_analysis(handle, trans, m, nnzL, descr,
                             d_csrValL, d_csrRowPtrL, d_csrColIdx,
                             info, policy, pBuffer);
        cudaDeviceSynchronize();

    }
    gettimeofday(&t2, NULL);
    double time_cuda_analysis = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
    time_cuda_analysis /= BENCH_REPEAT;
    *pre_time_add=time_cuda_analysis;
    
//    printf("cudaSPARSE SpTS analysis with layer on L used %4.2f ms\n", time_cuda_analysis);
    
    // L has unit diagonal, so no structural zero is reported.
    int structural_zero;
    status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
    }
    
    
    // step 5: solve L*y = x
    const double alpha = 1.;
    
    double dataSize = (double)((n+1)*sizeof(int) + nnzL*sizeof(int) + nnzL*sizeof(VALUE_TYPE) + 2*n*sizeof(VALUE_TYPE));
//    gettimeofday(&t1, NULL);
//    for(i=0;i<BENCH_REPEAT;i++)
//    {
//        cusparseScsrsv2_solve(handle, trans, m, nnzL, &alpha, descr,
//                          d_csrValL, d_csrRowPtrL, d_csrColIdx, info,
//                          d_b, d_x, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);
//        cudaDeviceSynchronize();
//    }
//    gettimeofday(&t2, NULL);
//    double time_cuda_solve = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
//    time_cuda_solve /= BENCH_REPEAT;
//    printf("cudaSPARSE SpTS without layer solve used %4.2f ms, throughput is %4.2f gflops, the bandwidth is %4.2f GB/s \n",
//           time_cuda_solve, 2*nnzL/(1e6*time_cuda_solve), dataSize/(1e6*time_cuda_solve));
    
//    *solve_time_add=time_cuda_solve;
//    *gflops_add=2*nnzL/(1e6*time_cuda_solve);
//    *bandwidth=dataSize/(1e6*time_cuda_solve);
    
    
    gettimeofday(&t1, NULL);
    for(i=0;i<BENCH_REPEAT;i++)
    {
        cusparseDcsrsv2_solve(handle, trans, m, nnzL, &alpha, descr,
                              d_csrValL, d_csrRowPtrL, d_csrColIdx, info,
                              d_b, d_x, policy, pBuffer);
        cudaDeviceSynchronize();
    }
    gettimeofday(&t2, NULL);
    double time_cuda_solve = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_solve /= BENCH_REPEAT;
//    printf("cudaSPARSE SpTS with layer solve used %4.2f ms, throughput is %4.2f gflops, the bandwidth is %4.2f GB/s\n",
//           time_cuda_solve, 2*nnzL/(1e6*time_cuda_solve), dataSize/(1e6*time_cuda_solve));
//
    *solve_time_add=time_cuda_solve;
    *gflops_add=2*nnzL/(1e6*time_cuda_solve);
    *bandwidth_add=dataSize/(1e6*time_cuda_solve);
    
    // L has unit diagonal, so no numerical zero is reported.
    int numerical_zero;
    status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }
    
    cudaMemcpy(x, d_x, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    // validate x
    int err_counter = 0;
    for (int i = 0; i < n; i++)
    {
        if (abs(x_ref[i] - x[i]) > 0.01 * abs(x_ref[i]))
            err_counter++;
    }
    if (!err_counter)
        //printf("cuda syncfree SpTS on L passed!\n");
        ;
    else
    {
        printf("cuda syncfree SpTS on L failed!\n");
        *solve_time_add=-1;
        *gflops_add=0;
        *bandwidth_add=0;
    }
    
    // step 6: free resources
    cudaFree(pBuffer);
    cusparseDestroyCsrsv2Info(info);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    
    free(x);
    free(x_ref);
    free(b);
    
    cudaFree(d_csrRowPtrL);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrValL);
    cudaFree(d_b);
    cudaFree(d_x);
    
    
    return 0;
}

#endif

