#include "common.h"
#include "mmio.h"
#include "read_mtx.h"
#include "tranpose.h"
#include "spts_cuSPARSE.h"



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

    //printf("name,nnz,layer,parallelism,pre_time(ms),solve_time,gflops,bandwith(GB/s)\n");

    
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    if(m==0 || nnzL==0)
        return -3;

    int layer;
    double parallelism,pre_time=0,solve_time,gflops,bandwith;

    matrix_layer(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp,&layer,&parallelism);

    //variance_in_warp(m,n,nnzL,csrRowPtrL_tmp,csrColIdxL_tmp,csrValL_tmp);
    
    

    spts_cuSPARSE(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL,
                        &pre_time, &solve_time, &gflops, &bandwith);

spts_cuSPARSE(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL,
&pre_time, &solve_time, &gflops, &bandwith);


//    fprintf(fp,"%s, %d, %d, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f\n",filename,nnzL,layer,parallelism,pre_time,solve_time,gflops,bandwith);
    fprintf(fp,"filename,m,nnzL,nnzL/m,layer,parallelism,pre_time,solve_time,gflops,bandwith\n");
    fprintf(fp,"%s, %d, %d, %4.2f, %d, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f\n",filename,m,nnzL,(double)nnzL/m,layer,parallelism,pre_time,solve_time,gflops,bandwith);

    

    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);
    fclose(fp);
    
    return 0;
}
    

