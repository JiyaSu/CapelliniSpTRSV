#include "common.h"
#include "mmio.h"
#include "read_mtx.h"
#include "tranpose.h"
#include "spts_syncfree_cuda2.h"



int main(int argc, char ** argv)
{

    FILE *fp = fopen("result.csv","a");
    if(fp==NULL)
        return -1;



    // report precision of floating-point
    //printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }
    
    //printf("PRECISION = %s\n", precision);
    //printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    //printf("---------------------------------------------------------------------------------------------\n");
    
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;
    
    //ex: ./spmv webbase-1M.mtx
    int argi = 1;
    
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    if(filename[strlen(filename)-1]==13)
        filename[strlen(filename)-1]='\0';
    //printf("-------------- %s --------------\n", filename);
    
    read_mtx(filename, &m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &csrValA);

    //printf("read_mtx finish\n");
    
    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int *csrRowPtrL_tmp ;
    int *csrColIdxL_tmp ;
    VALUE_TYPE *csrValL_tmp;


    if(m<=n)
        n=m;
    else
        m=n;
    if (m<=1)
        return 0;


    
    change2tran(m, nnzA,csrRowPtrA, csrColIdxA, csrValA, &nnzL, &csrRowPtrL_tmp, &csrColIdxL_tmp, &csrValL_tmp);
    //printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);



    
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    if(m==0 || nnzL==0)
        return -3;


    int *cscRowIdxTR = (int *)malloc(nnzL * sizeof(int));
    int *cscColPtrTR = (int *)malloc((n+1) * sizeof(int));
    memset(cscColPtrTR, 0, (n+1) * sizeof(int));
    VALUE_TYPE *cscValTR    = (VALUE_TYPE *)malloc(nnzL * sizeof(VALUE_TYPE));


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

//    for (int i = 0; i < BENCH_REPEAT; i++)
//    {
        matrix_transposition(m, n, nnzL,
                        csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp,
                        cscRowIdxTR, cscColPtrTR, cscValTR);
//    }

//    gettimeofday(&t2, NULL);
//    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
//    time_cuda_analysis /= BENCH_REPEAT;

//    printf("csr to csc used %4.2f ms\n", time_cuda_analysis);

    int layer;
    double parallelism,solve_time,gflops,bandwith,pre_time=0;

    matrix_layer(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp,&layer,&parallelism);

    //variance_in_warp(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp);
    
    
    VALUE_TYPE *x_ref;
    VALUE_TYPE *b ;

    get_x_b(m, n, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, &x_ref, &b);

    VALUE_TYPE *x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);


    sptrsv_syncfree_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzL,x,b,x_ref,&pre_time,&solve_time,&gflops,&bandwith);
    sptrsv_syncfree_cuda(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzL,x,b,x_ref,&pre_time,&solve_time,&gflops,&bandwith);

//    spts_syncfree_cuda2(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL,
//        &solve_time, &gflops, &bandwith);

    //fprintf(fp,"%s, %d, %d, %4.2f, %4.2f, %4.2f, %4.2f\n",filename,nnzL,layer,parallelism,solve_time,gflops,bandwith);
    fprintf(fp,"%s, %d, %d, %4.2f, %d, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f\n",filename,m,nnzL,(double)nnzL/m,layer,parallelism,pre_time,solve_time,gflops,bandwith);

    

    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);
    fclose(fp);

    free(cscRowIdxTR);
    free(cscColPtrTR);
    free(cscValTR);
    free(x_ref);
    free(b);
    free(x);

    return 0;
}
    

