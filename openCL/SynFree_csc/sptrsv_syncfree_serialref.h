#ifndef _SPTRSV_SYNCFREE_SERIALREF_
#define _SPTRSV_SYNCFREE_SERIALREF_

#include "common.h"

int sptrsv_syncfree_analyser(const int   *cscRowIdx,
                             const int    m,
                             const int    n,
                             const int    nnz,
                                   int   *csrRowHisto)
{
    memset(csrRowHisto, 0, m * sizeof(int));

    // generate row pointer by partial transposition
//#pragma omp parallel for
    for (int i = 0; i < nnz; i++)
    {
//#pragma omp atomic
        csrRowHisto[cscRowIdx[i]]++;
    }

    return 0;
}

int sptrsv_syncfree_executor(const int           *cscColPtr,
                             const int           *cscRowIdx,
                             const VALUE_TYPE    *cscVal,
                             const int           *graphInDegree,
                             const int            m,
                             const int            n,
                             const int            substitution,
                             const int            rhs,
                             const VALUE_TYPE    *b,
                                   VALUE_TYPE    *x)
{
    // malloc tmp memory to simulate atomic operations
    int *graphInDegree_atomic = (int *)malloc(sizeof(int) * m);
    memset(graphInDegree_atomic, 0, sizeof(int) * m);

    // malloc tmp memory to collect a partial sum of each row
    VALUE_TYPE *left_sum = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
    memset(left_sum, 0, sizeof(VALUE_TYPE) * m * rhs);

    if (substitution == SUBSTITUTION_FORWARD)
    {
        for (int i = 0; i < n; i++)
        {
            int dia = graphInDegree[i] - 1;

            // while loop, i.e., wait, until all nnzs are prepared
            do
            {
                // just wait
            }
            while (dia != graphInDegree_atomic[i]);

            for (int k = 0; k < rhs; k++)
            {
                VALUE_TYPE xi = (b[i * rhs + k] - left_sum[i * rhs + k]) / cscVal[cscColPtr[i]];
                x[i * rhs + k] = xi;
            }

            for (int j = cscColPtr[i] + 1; j < cscColPtr[i+1]; j++)
            {
                int rowIdx = cscRowIdx[j];
                // atomic add
                for (int k = 0; k < rhs; k++)
                    left_sum[rowIdx * rhs + k] += x[i * rhs + k] * cscVal[j];
                graphInDegree_atomic[rowIdx] += 1;
            }
        }
    }
    else if (substitution == SUBSTITUTION_BACKWARD)
    {
        for (int i = n-1; i >= 0; i--)
        {
            int dia = graphInDegree[i] - 1;

            // while loop, i.e., wait, until all nnzs are prepared
            do
            {
                // just wait
            }
            while (dia != graphInDegree_atomic[i]);

            for (int k = 0; k < rhs; k++)
            {
                VALUE_TYPE xi = (b[i * rhs + k] - left_sum[i * rhs + k]) / cscVal[cscColPtr[i+1]-1];
                x[i * rhs + k] = xi;
            }

            for (int j = cscColPtr[i]; j < cscColPtr[i+1]-1; j++)
            {
                int rowIdx = cscRowIdx[j];
                // atomic add
                for (int k = 0; k < rhs; k++)
                    left_sum[rowIdx * rhs + k] += x[i * rhs + k] * cscVal[j];
                graphInDegree_atomic[rowIdx] += 1;
                //printf("node %i updated node %i\n", i, rowIdx);
            }
        }
    }

    free(graphInDegree_atomic);
    free(left_sum);

    return 0;
}

int sptrsv_syncfree_serialref(const int           *cscColPtrTR,
                              const int           *cscRowIdxTR,
                              const VALUE_TYPE    *cscValTR,
                              const int            m,
                              const int            n,
                              const int            nnzTR,
                              const int            substitution,
                              const int            rhs,
                                    VALUE_TYPE    *x,
                              const VALUE_TYPE    *b,
                              const VALUE_TYPE    *x_ref)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    //  - SpTRSV Serial analyser start!
    printf(" - SpTRSV Serial analyser start!\n");

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    int *graphInDegree = (int *)malloc(m * sizeof(int));

    sptrsv_syncfree_analyser(cscRowIdxTR, m, n, nnzTR, graphInDegree);

    gettimeofday(&t2, NULL);
    double time_sptrsv_analyser = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("SpTRSV Serial analyser on L used %4.2f ms\n", time_sptrsv_analyser);

    //  - SpTRSV Serial executor start!
    printf(" - SpTRSV Serial executor start!\n");

    gettimeofday(&t1, NULL);

    sptrsv_syncfree_executor(cscColPtrTR, cscRowIdxTR, cscValTR,
                             graphInDegree, m, n, substitution, rhs, b, x);

    gettimeofday(&t2, NULL);
    double time_sptrsv_executor = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    double flop = 2*(double)rhs*(double)nnzTR;
    printf("SpTRSV Serial executor used %4.2f ms, throughput is %4.2f gflops\n",
           time_sptrsv_executor, flop/(1e6*time_sptrsv_executor));

    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("SpTRSV Serial executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("SpTRSV Serial executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    free(graphInDegree);

    return 0;
}

#endif
