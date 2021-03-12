#ifndef _TRANS_
#define _TRANS_

#include "common.h"


typedef struct node
{
    int layer;
    int i;
}node;



// in-place exclusive scan
void exclusive_scan(int *input, int length)
{
    if(length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    int i;
    for (i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}



void matrix_transposition(const int         m,
                          const int         n,
                          const int         nnz,
                          const int        *csrRowPtr,
                          const int        *csrColIdx,
                          const VALUE_TYPE *csrVal,
                          int        *cscRowIdx,
                          int        *cscColPtr,
                          VALUE_TYPE *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(int) * (n+1));
    int i;
    for (i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    int *cscColIncr = (int *)malloc(sizeof(int) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(int) * (n+1));

    // insert nnz to csc
    int row,j;
    for (row = 0; row < m; row++)
    {
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}


int matrix_layer(const int         m,
                 const int         n,
                 const int         nnz,
                 const int        *csrRowPtr,
                 const int        *csrColIdx,
                 const VALUE_TYPE *csrVal,
                 int              *layer_add,
                 double           *parallelism_add
                 )

{
    int *layer=(int *)malloc(m*sizeof(int));
    if (layer==NULL)
        printf("layer error\n");
    memset (layer, 0, sizeof(int)*m);

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    int max_layer;
    int max_layer2=0;
    int max=0;
    unsigned int min=-1;

    // count layer
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((layer[col]+1)>max_layer)
                max_layer=layer[col]+1;

        }
        layer[row]=max_layer;
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    for(j=1;j<=max_layer2;j++)
    {
        if(max<layer_num[j])
            max=layer_num[j];
        if(min>layer_num[j])
            min=layer_num[j];
    }

    double avg=(double)m/max_layer2;
    free(layer);
    free(layer_num);

    //printf("matrix L's layer = %d, average numer of nodes in layer = %d\n",max_layer2,avg);
    int min2=min;
    //printf("the minimun parallelism is %d,the maximun parallelism is %d\n",min2,max);
    *layer_add=max_layer2;
    *parallelism_add=avg;
    //printf(",%d,%d,%d",nnz,max_layer2,avg);
    return max_layer2;

}

void matrix_transposition_with_layer(const int         m,
                                     const int         n,
                                     const int         nnz,
                                     const int        *csrRowPtr,
                                     const int        *csrColIdx,
                                     const VALUE_TYPE *csrVal,
                                     int* order,
                                     int* *layer_num_add,
                                     int *layers,
                                     int *max_layer_num)
{

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));

    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);

    int i;

    int max_layer;
    int max_layer2=0;

    // insert nnz to csc
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((row_layer[col].layer+1)>max_layer)
                max_layer=row_layer[col].layer+1;

        }
        row_layer[row].layer=max_layer;
        row_layer[row].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    //printf("csc over\n");
    //printf("layer is %d\n",max_layer2);

    //int layer_num2[max_layer2];
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));

    *layers=max_layer2;
    *layer_num_add=layer_num2;

    int sum=0;
    int n_layer=0;
    //printf("%d %d\n",max_layer2+1,m);
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        if(n_layer<layer_num[i])
            n_layer=layer_num[i];
        sum+=layer_num[i];
        layer_num2[i]=sum;
    }
    *max_layer_num=n_layer;
    //printf("\n");

    //printf("layer=%d\n",max_layer2);

    int tmp_n;
    for (row = 0; row < m; row++)
    {
        tmp_n=layer_num2[row_layer[row].layer-1]+row_layer[row].i;
        order[tmp_n]=row;
        //order2[row]=tmp_n;
    }

    //    printf("begin free\n");
    //    printf("22202020002\n");


    free(layer_num);
    layer_num=NULL;
    //    printf("free layer_num\n");

    free(row_layer);
    row_layer=NULL;
    //    printf("free row_layer\n");
}

void matrix_transposition_with_layer2(const int         m,
                                     const int         n,
                                     const int         nnz,
                                     const int        *csrRowPtr,
                                     const int        *csrColIdx,
                                     const VALUE_TYPE *csrVal,
                                     int**   order_add,
                                     int*    order_len_add
                                     )
{

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));

    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);

    int i;
    int max_layer;
    int max_layer2=0;

    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]-1; j++)
        {
            int col = csrColIdx[j];
            
            if((row_layer[col].layer+1)>max_layer)
            {
                max_layer=row_layer[col].layer+1;
            }

        }
        row_layer[row].layer=max_layer;
        row_layer[row].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }

    max_layer2++;
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));

    int sum=0;
    int tmp_layer_num;

    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        tmp_layer_num = ceil((double)layer_num[i-1]/(double)WARP_SIZE)*WARP_SIZE;
        sum+=tmp_layer_num;
        layer_num2[i]=sum;
    }


//        for (i=0;i<max_layer2+1;i++)
//            printf("%d ",layer_num2[i]);
//        printf("\n");
//    //
    //    for (i=0;i<max_layer2;i++)
    //        printf("%d ",layer_num2[i+1]-layer_num2[i]);
    //    printf("\n");

    int * order = (int *)malloc(sum*sizeof(int));
    if (order==NULL)
        printf("order error\n");
    memset (order, -1, sizeof(int)*sum);


    int tmp_n;
    for (row = 0; row < m; row++)
    {
        tmp_n=layer_num2[row_layer[row].layer]+row_layer[row].i;
        order[tmp_n]=row;
        
    }

    *order_add=order;
    *order_len_add=sum;






    //printf("begin free\n");

    free(layer_num);
    layer_num=NULL;
    //printf("free layer_num\n");

    free(row_layer);
    row_layer=NULL;
    //printf("free row_layer\n");

    free(layer_num2);
    layer_num2=NULL;
    //printf("free layer_num2\n");

}

void matrix_transposition_with_layer3(const int         m,
                                      const int         n,
                                      const int         nnz,
                                      const int        *csrRowPtr,
                                      const int        *csrColIdx,
                                      const VALUE_TYPE *csrVal,
                                      int**   order_add,
                                      int*    order_len_add
                                      )
{
    
    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);
    
    int i;
    int max_layer;
    int max_layer2=0;
    
    int row,j,row2;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        row2=csrColIdx[csrRowPtr[row+1]-1];
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]-1; j++)
        {
            int col = csrColIdx[j];
            
            if((row_layer[col].layer+1)>max_layer)
            {
                max_layer=row_layer[col].layer+1;
            }
            
        }
        row_layer[row2].layer=max_layer;
        row_layer[row2].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    
    max_layer2++;
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));
    
    int sum=0;
    int tmp_layer_num;
    
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        tmp_layer_num = ceil((double)layer_num[i-1]/(double)WARP_SIZE)*WARP_SIZE;
        sum+=tmp_layer_num;
        layer_num2[i]=sum;
    }
    
    
    //        for (i=0;i<max_layer2+1;i++)
    //            printf("%d ",layer_num2[i]);
    //        printf("\n");
    //    //
    //    for (i=0;i<max_layer2;i++)
    //        printf("%d ",layer_num2[i+1]-layer_num2[i]);
    //    printf("\n");
    
    int * order = (int *)malloc(sum*sizeof(int));
    if (order==NULL)
        printf("order error\n");
    memset (order, -1, sizeof(int)*sum);
    
    
    int tmp_n;
    for (row = 0; row < m; row++)
    {
        row2=csrColIdx[csrRowPtr[row+1]-1];
        tmp_n=layer_num2[row_layer[row2].layer]+row_layer[row2].i;
        order[tmp_n]=row;
        
    }
    
    *order_add=order;
    *order_len_add=sum;
    
    
    
    
    
    
    //printf("begin free\n");
    
    free(layer_num);
    layer_num=NULL;
    //printf("free layer_num\n");
    
    free(row_layer);
    row_layer=NULL;
    //printf("free row_layer\n");
    
    free(layer_num2);
    layer_num2=NULL;
    //printf("free layer_num2\n");
    
}

void matrix_transposition_with_layer4(const int         m,
                                      const int         n,
                                      const int         nnz,
                                      const int        *csrRowPtr,
                                      const int        *csrColIdx,
                                      const VALUE_TYPE *csrVal,
                                      int**   order_add,
                                      int*    order_len_add
                                      )
{
    
    int *layer=(int *)malloc((m+1)*sizeof(int));
    if (layer==NULL)
        printf("layer_num error\n");
    memset (layer, 0, sizeof(int)*(m+1));
    
//    node *row_layer = (node *)malloc(sizeof(node)*m);
//    if (row_layer==NULL)
//        printf("row_layer error\n");
//    memset (row_layer, 0, sizeof(node)*m);
    int sum=2*m;
    int * order = (int *)malloc(sum*sizeof(int));
    if (order==NULL)
        printf("order error\n");
    memset (order, -1, sizeof(int)*sum);
    
    int i;
    int max_layer=0;
    
    int num_in_layer=0;
    int start=0;
    int start_n=0;
    
    int row,j,row2,k;
    for (row = 0; row < m; row++)
    {
        row2=csrColIdx[csrRowPtr[row+1]-1];
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]-1; j++)
        {
            int col = csrColIdx[j];
            
            if(layer[col]==max_layer)
            {
                max_layer++;
                for(k=0;k<(num_in_layer);k++)
                    order[k+start]=start_n+k;
                
                start+=ceil((double)num_in_layer/(double)WARP_SIZE)*WARP_SIZE;
                start_n+=num_in_layer;
                num_in_layer=0;
                break;
            }
            
        }
        layer[row2]=max_layer;
        num_in_layer++;
    }
    for(k=0;k<(num_in_layer);k++)
        order[k+start]=start_n+k;
    
    start+=ceil((double)num_in_layer/(double)WARP_SIZE)*WARP_SIZE;
    start_n+=num_in_layer;
    
    *order_add=order;
    *order_len_add=start;
    
//    max_layer2++;
//    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
//    if (layer_num2==NULL)
//        printf("layer_num2 error\n");
//    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));
//
//    int sum=0;
//    int tmp_layer_num;
//
//    for (i=1; i<=max_layer2; i++)
//    {
//        //printf("%d ",layer_num[i]);
//        tmp_layer_num = ceil((double)layer_num[i-1]/(double)WARP_SIZE)*WARP_SIZE;
//        sum+=tmp_layer_num;
//        layer_num2[i]=sum;
//    }
//
//
//    //        for (i=0;i<max_layer2+1;i++)
//    //            printf("%d ",layer_num2[i]);
//    //        printf("\n");
//    //    //
//    //    for (i=0;i<max_layer2;i++)
//    //        printf("%d ",layer_num2[i+1]-layer_num2[i]);
//    //    printf("\n");
//
//    int * order = (int *)malloc(sum*sizeof(int));
//    if (order==NULL)
//        printf("order error\n");
//    memset (order, -1, sizeof(int)*sum);
//
//
//    int tmp_n;
//    for (row = 0; row < m; row++)
//    {
//        row2=csrColIdx[csrRowPtr[row+1]-1];
//        tmp_n=layer_num2[row_layer[row2].layer]+row_layer[row2].i;
//        order[tmp_n]=row;
//
//    }
//
//    *order_add=order;
//    *order_len_add=sum;
//
    
    
    
    
    
    //printf("begin free\n");
    
    free(layer);
    layer=NULL;
    //printf("free layer_num\n");
//
//    free(row_layer);
//    row_layer=NULL;
//    //printf("free row_layer\n");
//
//    free(layer_num2);
//    layer_num2=NULL;
//    //printf("free layer_num2\n");
    
}



void matrix_transposition_to_layer_csr(const int         m,
                                     const int         n,
                                     const int         nnz,
                                     const int        *csrRowPtr,
                                     const int        *csrColIdx,
                                     const VALUE_TYPE *csrVal,
                                     int              *RowPtr,
                                     int              *ColIdx,
                                     VALUE_TYPE       *Val
                                       )

{
    memset (RowPtr, 0, sizeof(int)*(m+1));
    memset (ColIdx, 0, sizeof(int)*(nnz));
    memset (Val, 0, sizeof(VALUE_TYPE)*(nnz));

    int* order=(int *)malloc((m)*sizeof(int));
    if(order==NULL)
        printf("order error\n");
    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));

    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);

    int i;

    int max_layer;
    int max_layer2=0;

    // insert nnz to csc
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((row_layer[col].layer+1)>max_layer)
                max_layer=row_layer[col].layer+1;

        }
        row_layer[row].layer=max_layer;
        row_layer[row].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    //printf("csc over\n");
    //printf("layer is %d\n",max_layer2);

    //int layer_num2[max_layer2];
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));


    int sum=0;
    int n_layer=0;
    //printf("%d %d\n",max_layer2+1,m);
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
//        if(n_layer<layer_num[i])
//            n_layer=layer_num[i];
        sum+=layer_num[i];
        layer_num2[i]=sum;
    }

    //printf("\n");

    //printf("layer=%d\n",max_layer2);

    int tmp_n;
    for (row = 0; row < m; row++)
    {
        tmp_n=layer_num2[row_layer[row].layer-1]+row_layer[row].i;
        order[tmp_n]=row;
        //order2[row]=tmp_n;
    }

    sum=0;
    int Len=0;
    int base1,base2;
    for (i=0;i<m;i++)
    {
        n_layer=order[i];
        Len=csrRowPtr[n_layer+1]-csrRowPtr[n_layer];
        sum+=Len;
        RowPtr[i+1]=sum;
        base1=RowPtr[i];
        base2=csrRowPtr[n_layer];
        for(j=0;j<Len;j++)
        {
            ColIdx[base1+j]=csrColIdx[base2+j];
            Val[base1+j]=csrVal[base2+j];
        }
    }


    free(layer_num);
    layer_num=NULL;
    //    printf("free layer_num\n");

    free(row_layer);
    row_layer=NULL;
    //    printf("free row_layer\n");

    free(order);
    order=NULL;
}

void matrix_transposition_to_layer_csr2(const int         m,
                                       const int         n,
                                       const int         nnz,
                                       const int        *csrRowPtr,
                                       const int        *csrColIdx,
                                       const VALUE_TYPE *csrVal,
                                       int              *RowPtr,
                                       int              *ColIdx,
                                       VALUE_TYPE       *Val
                                       )

{
    memset (RowPtr, 0, sizeof(int)*(m+1));
    memset (ColIdx, 0, sizeof(int)*(nnz));
    memset (Val, 0, sizeof(VALUE_TYPE)*(nnz));
    
    int* order=(int *)malloc((m)*sizeof(int));
    if(order==NULL)
        printf("order error\n");
    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);
    
    int i;
    
    int max_layer;
    int max_layer2=0;
    
    // insert nnz to csc
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]-1; j++)
        {
            int col = csrColIdx[j];
            
            if((row_layer[col].layer+1)>max_layer)
                max_layer=row_layer[col].layer+1;
            
        }
        row_layer[row].layer=max_layer;
        row_layer[row].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    
    max_layer2++;
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));
    
    int sum=0;
    int tmp_layer_num;
    
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        sum+=layer_num[i-1];
        layer_num2[i]=sum;
        
    }
    if(sum!=m)
        printf("error\n");
    
    
    //        for (i=0;i<max_layer2+1;i++)
    //            printf("%d ",layer_num2[i]);
    //        printf("\n");
    //    //
    //    for (i=0;i<max_layer2;i++)
    //        printf("%d ",layer_num2[i+1]-layer_num2[i]);
    //    printf("\n");
    
    
    
    
    int tmp_n;
    for (row = 0; row < m; row++)
    {
        tmp_n=layer_num2[row_layer[row].layer]+row_layer[row].i;
        order[tmp_n]=row;
        
    }
    
    sum=0;
    int Len=0;
    int base1,base2,n_layer;
    for (i=0;i<m;i++)
    {
        n_layer=order[i];
        Len=csrRowPtr[n_layer+1]-csrRowPtr[n_layer];
        sum+=Len;
        RowPtr[i+1]=sum;
        base1=RowPtr[i];
        base2=csrRowPtr[n_layer];
        for(j=0;j<Len;j++)
        {
            ColIdx[base1+j]=csrColIdx[base2+j];
            Val[base1+j]=csrVal[base2+j];
        }
    }
    
    for(i=0;i<m;i++)
    {
        int row=ColIdx[RowPtr[i+1]-1];
        if(order[i]!=row)
        {
            printf("ee %d %d\n",order[i],row);
        }
//        if(i<64)
//            printf("%d %d\n",order[i],row);
    }
    
    free(layer_num);
    layer_num=NULL;
    //    printf("free layer_num\n");
    
    free(row_layer);
    row_layer=NULL;
    //    printf("free row_layer\n");
    
    free(order);
    order=NULL;
}

void matrix_transposition_and_layer2(const int         m,
                                    const int         n,
                                    const int         nnz,
                                    const int        *csrRowPtr,
                                    const int        *csrColIdx,
                                    const VALUE_TYPE *csrVal,
                                    int        *cscRowIdx,
                                    int        *cscColPtr,
                                    VALUE_TYPE *cscVal,
                                    int* order,
                                    int* *layer_num_add,
                                    int *layers,
                                    int *max_layer_num)
{
    // histogram in column pointer
    //printf("matrix_transposition begin\n");
    memset (cscColPtr, 0, sizeof(int) * (n+1));

    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));

    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);

    int i;

    for (i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    //printf("xclusive_scan begin\n");
    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    int *cscColIncr = (int *)malloc(sizeof(int) * (n+1));
    if (cscColIncr==NULL)
        printf("cscColIncr error\n");
    memcpy (cscColIncr, cscColPtr, sizeof(int) * (n+1));
    int max_layer;
    int max_layer2=0;

    // insert nnz to csc
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            if((row_layer[col].layer+1)>max_layer)
                max_layer=row_layer[col].layer+1;

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
        row_layer[row].layer=max_layer;
        row_layer[row].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    //printf("csc over\n");
    //printf("layer is %d\n",max_layer2);

    //int layer_num2[max_layer2];
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));

    *layers=max_layer2;
    *layer_num_add=layer_num2;

    int sum=0;
    int n_layer=0;
    //printf("%d %d\n",max_layer2+1,m);
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        if(n_layer<layer_num[i])
            n_layer=layer_num[i];
        sum+=layer_num[i];
        layer_num2[i]=sum;
    }
    *max_layer_num=n_layer;
    //printf("\n");

    //printf("layer=%d\n",max_layer2);

    int tmp_n;
    for (row = 0; row < m; row++)
    {
        tmp_n=layer_num2[row_layer[row].layer-1]+row_layer[row].i;
        order[tmp_n]=row;
        //order2[row]=tmp_n;
    }

//    printf("begin free\n");
//    printf("22202020002\n");


    free(layer_num);
    layer_num=NULL;
//    printf("free layer_num\n");

    free(row_layer);
    row_layer=NULL;
//    printf("free row_layer\n");

//    free(layer_num2);
//    layer_num2=NULL;
//    printf("free layer_num2\n");

    free (cscColIncr);
    cscColIncr=NULL;
//    printf("free cscColIncr\n");
}

void matrix_layer2                   (const int         m,
                                     const int         n,
                                     const int         nnz,
                                     const int        *csrRowPtr,
                                     const int        *csrColIdx,
                                     const VALUE_TYPE *csrVal,
                                     int* *layer_num_add,
                                     int *layers)
{
    
    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);
    
    int i;
    int max_layer;
    int max_layer2=0;
    
    // insert nnz to csc
    int row,j,row2;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        row2=csrColIdx[csrRowPtr[row+1]-1];
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]-1; j++)
        {
            int col = csrColIdx[j];
            
            if((row_layer[col].layer+1)>max_layer)
            {
                max_layer=row_layer[col].layer+1;
            }
            
        }
        row_layer[row2].layer=max_layer;
        row_layer[row2].i=layer_num[max_layer];
        layer_num[max_layer]++;
//        if(max_layer==0)
//            printf("%d %d\n",row2,layer_num[max_layer]);
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    //printf("csc over\n");
    //printf("layer is %d\n",max_layer2);
    
    //int layer_num2[max_layer2];
    max_layer2++;
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));
    
    *layers=max_layer2;
    *layer_num_add=layer_num2;
    
    int sum=0;
    int n_layer=0;
    //printf("%d %d\n",max_layer2+1,m);
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        
        sum+=layer_num[i-1];
        layer_num2[i]=sum;
    }
   
    free(layer_num);
    layer_num=NULL;
    //    printf("free layer_num\n");
    
    free(row_layer);
    row_layer=NULL;
    //    printf("free row_layer\n");
    
    //    free(layer_num2);
    //    layer_num2=NULL;
    //    printf("free layer_num2\n");
    
//    free (cscColIncr);
//    cscColIncr=NULL;
    //    printf("free cscColIncr\n");
}

void matrix_layer3                   (const int         m,
                                      const int         n,
                                      const int         nnz,
                                      const int        *csrRowPtr,
                                      const int        *csrColIdx,
                                      const VALUE_TYPE *csrVal,
                                      int* *layer_num_add,
                                      int *layers)
{
    
    int *layer=(int *)malloc((m+1)*sizeof(int));
    if (layer==NULL)
        printf("layer error\n");
    memset (layer, 0, sizeof(int)*(m+1));
    
//    node *row_layer = (node *)malloc(sizeof(node)*m);
//    if (row_layer==NULL)
//        printf("row_layer error\n");
//    memset (row_layer, 0, sizeof(node)*m);
    
    int *layer_num2=(int *)malloc((m+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(m+1));
    
    int i;
    int max_layer=0;
    int nn=0;
    
    // insert nnz to csc
    int row,j,row2;
    for (row = 0; row < m; row++)
    {
        
        row2=csrColIdx[csrRowPtr[row+1]-1];
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]-1; j++)
        {
            int col = csrColIdx[j];
            
            if(layer[col]==max_layer)
            {
                max_layer++;
                layer_num2[max_layer]=nn;
                break;
            }
            
        }
        layer[row2]=max_layer;
        nn++;
    }
    //printf("csc over\n");
    //printf("layer is %d\n",max_layer2);
    
    //int layer_num2[max_layer2];
    max_layer++;
    layer_num2[max_layer]=nn;
    
    *layers=max_layer;
    *layer_num_add=layer_num2;
    
//    int sum=0;
//    int n_layer=0;
//    //printf("%d %d\n",max_layer2+1,m);
//    for (i=1; i<=max_layer2; i++)
//    {
//        //printf("%d ",layer_num[i]);
//
//        sum+=layer_num[i-1];
//        layer_num2[i]=sum;
//    }
    
    free(layer);
    layer=NULL;
    //    printf("free layer_num\n");
    
//    free(row_layer);
//    row_layer=NULL;
    //    printf("free row_layer\n");
    
    //    free(layer_num2);
    //    layer_num2=NULL;
    //    printf("free layer_num2\n");
    
    //    free (cscColIncr);
    //    cscColIncr=NULL;
    //    printf("free cscColIncr\n");
}


void variance_in_warp(const int         m,
                        const int         n,
                        const int         nnz,
                        const int        *csrRowPtr,
                        const int        *csrColIdx,
                        const VALUE_TYPE *csrVal)
{
    int *layer_num=(int *)malloc((m+1)*sizeof(int));
    if (layer_num==NULL)
        printf("layer_num error\n");
    memset (layer_num, 0, sizeof(int)*(m+1));
    
    node *row_layer = (node *)malloc(sizeof(node)*m);
    if (row_layer==NULL)
        printf("row_layer error\n");
    memset (row_layer, 0, sizeof(node)*m);
    
    int *order=(int *)malloc((m)*sizeof(int));
    if (order==NULL)
        printf("order error\n");
    memset (order, 0, sizeof(int)*(m));
    
    int i;
    
    int max_layer;
    int max_layer2=0;
    
    
    int row,j;
    for (row = 0; row < m; row++)
    {
        max_layer=0;
        for (j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];
            
            if((row_layer[col].layer+1)>max_layer)
                max_layer=row_layer[col].layer+1;
            
        }
        row_layer[row].layer=max_layer;
        row_layer[row].i=layer_num[max_layer];
        layer_num[max_layer]++;
        if (max_layer>max_layer2)
            max_layer2=max_layer;
    }
    
    int *layer_num2=(int *)malloc((max_layer2+1)*sizeof(int));
    if (layer_num2==NULL)
        printf("layer_num2 error\n");
    memset (layer_num2, 0, sizeof(int)*(max_layer2+1));
    
    
    
    int sum=0;
    int n_layer=0;
    //printf("%d %d\n",max_layer2+1,m);
    for (i=1; i<=max_layer2; i++)
    {
        //printf("%d ",layer_num[i]);
        if(n_layer<layer_num[i])
            n_layer=layer_num[i];
        sum+=layer_num[i];
        layer_num2[i]=sum;
    }
    //printf("\n");
    
    //printf("layer=%d\n",max_layer2);
    
    int tmp_n;
    for (row = 0; row < m; row++)
    {
        tmp_n=row_layer[row].layer-1;
        order[row]=tmp_n;
        //order2[row]=tmp_n;
    }
    
    double sum2=0,avg=0,delta;
    int nn;
    int warp=0;
    double max=0;
    double sum3=0;
    int warp3=0;
    for(i=0;i<m;i+=WARP_SIZE)
    {
        avg=0;
        for(j=i;j<(i+WARP_SIZE) && j<m;j++)
        {
            avg+=order[j];
        }
        nn=j-i;
        avg=avg/nn;
        //printf("%g ",avg);
        delta=0;
        for(j=i;j<(i+WARP_SIZE) && j<m;j++)
        {
            delta+=(order[j]-avg)*(order[j]-avg);
        }
        delta=delta/nn;
        //printf("%g ",delta);
        if(max<delta)
            max=delta;
        if(delta>0)
        {
            sum3+=delta;
            warp3++;
        }
        sum2+=delta;
        warp++;
    }
    //printf("\n");
    sum2=sum2/warp;
    sum3=sum3/warp3;
    printf("The average variance in warp is %g.\n",sum2);
    //printf("The average variance without zero in warp is %g.\n",sum3);
    printf("The max variance in warp is %g.\n",max);
    
    
    free(layer_num);
    layer_num=NULL;
    //    printf("free layer_num\n");
    
    free(row_layer);
    row_layer=NULL;
    
    free(layer_num2);
    layer_num2=NULL;
    
    free(order);
    order=NULL;
    return;
}

#endif

