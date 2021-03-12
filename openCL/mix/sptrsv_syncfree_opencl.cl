#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#define WARP_SIZE 64

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK   4
#endif

#define SUBSTITUTION_FORWARD 0
#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3

//inline
//void atom_add_d_fp32(volatile __global float *val,
//                   float delta)
//{
//    union { float f; unsigned int i; } old;
//    union { float f; unsigned int i; } new;
//    do
//    {
//        old.f = *val;
//        new.f = old.f + delta;
//    }
//    while (atomic_cmpxchg((volatile __global unsigned int *)val, old.i, new.i) != old.i);
//}
//
//inline
//void atom_add_d_fp64(volatile __global double *val,
//                   double delta)
//{
//    union { double f; ulong i; } old;
//    union { double f; ulong i; } new;
//    do
//    {
//        old.f = *val;
//        new.f = old.f + delta;
//    }
//    while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);
//}
//inline
//void atom_add_s_fp32(volatile __local float *val,
//                   float delta)
//{
//    union { float f; unsigned int i; } old;
//    union { float f; unsigned int i; } new;
//    do
//    {
//        old.f = *val;
//        new.f = old.f + delta;
//    }
//    while (atomic_cmpxchg((volatile __local unsigned int *)val, old.i, new.i) != old.i);
//}
//
//inline
//void atom_add_d_fp(volatile __global VALUE_TYPE *address,
//                   VALUE_TYPE val)
//{
//    if (sizeof(VALUE_TYPE) == 8)
//        atom_add_d_fp64(address, val);
//    else
//        atom_add_d_fp32(address, val);
//}
//
//inline
//void atom_add_s_fp64(volatile __local double *val,
//                   double delta)
//{
//    union { double f; ulong i; } old;
//    union { double f; ulong i; } new;
//    do
//    {
//        old.f = *val;
//        new.f = old.f + delta;
//    }
//    while (atom_cmpxchg((volatile __local ulong *)val, old.i, new.i) != old.i);
//}
//
//__kernel
//void sptrsv_syncfree_opencl_analyser(__global const int      *d_cscRowIdx,
//                                     const int                m,
//                                     const int                nnz,
//                                     __global int            *d_graphInDegree)
//{
//    const int global_id = get_global_id(0);
//    if (global_id < nnz)
//    {
//        atomic_fetch_add_explicit((atomic_int*)&d_graphInDegree[d_cscRowIdx[global_id]], 1,
//                                  memory_order_acq_rel, memory_scope_device);
//    }
//}

__kernel
void sptrsv_syncfree_opencl_executor(__global const int            *d_csrRowPtr,
                                     __global const int            *d_csrColIdx,
                                     __global const VALUE_TYPE     *d_csrVal,
                                     __global volatile int         *d_get_value,
                                     const int                      m,
                                     __global const VALUE_TYPE     *d_b,
                                     __global volatile VALUE_TYPE  *d_x,
                                     __global const int            *d_warp_num,
                                     const int                      Len)
{
    const int global_id = get_global_id(0);
    const int warp_id = global_id/WARP_SIZE;
    const int local_id = get_local_id(0);
    
    int row;
    
    if(warp_id>=(Len-1))
        return;
    
    const int lane_id = (WARP_SIZE - 1) & local_id;
    __local VALUE_TYPE s_left_sum[WARP_PER_BLOCK*WARP_SIZE];
    
    
    if(d_warp_num[warp_id+1]>(d_warp_num[warp_id]+1))
    {
        //thread
        row =d_warp_num[warp_id]+lane_id;
        if(row>=m)
            return;

        int col,j,i;
        VALUE_TYPE xi;
        VALUE_TYPE left_sum=0;
        i=row;
        j=d_csrRowPtr[i];

        while(j<d_csrRowPtr[i+1])
        {
            col=d_csrColIdx[j];
            if(atomic_load_explicit((atomic_int*)&d_get_value[col],memory_order_acquire, memory_scope_device)==1)
                //while(d_get_value[col]==1)
                //if(d_get_value[col]==1)
            {
                left_sum+=d_csrVal[j]*d_x[col];
                j++;
                col=d_csrColIdx[j];
            }
            if(i==col)
            {
                xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
                d_x[i] = xi;
                mem_fence(CLK_GLOBAL_MEM_FENCE);
                d_get_value[i]=1;
                j++;
            }
        }
    }
    else
    {
        //warp
        
        row = d_warp_num[warp_id];
        if(row>=m)
            return;

        int col,j=d_csrRowPtr[row]  + lane_id;
        VALUE_TYPE xi,sum=0;
        while(j < (d_csrRowPtr[row+1]-1))
        {
            col=d_csrColIdx[j];
            //if(d_get_value[col]==1)
            if(atomic_load_explicit((atomic_int*)&d_get_value[col],memory_order_acquire, memory_scope_device)==1)
            {
                sum += d_x[col] * d_csrVal[j];
                j += WARP_SIZE;
            }
        }

        s_left_sum[local_id]=sum;

        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        {
            if(lane_id < offset)
            {
                s_left_sum[local_id] += s_left_sum[local_id+offset];
            }
        }



        if (!lane_id)
        {
            xi = (d_b[row] - s_left_sum[local_id]) / d_csrVal[d_csrRowPtr[row+1]-1];
            d_x[row]=xi;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            d_get_value[row]=1;
            //        if(xi!=1)
            //            printf("%d %g %g %d\n",global_x_id,d_b[global_x_id],left_sum[local_warp_id],thread_in_warp[local_warp_id]);
            //        if(global_x_id==2197954)
            //            printf("%d %g\n",global_x_id,xi);
        }
    }
    
    
    
//    const int local_id = get_local_id(0);
//
//
//
//
//
//    int global_x_id = global_id / WARP_SIZE;
//    if (global_x_id >= m) return;
//
//
//
//    // Initialize
//    const int local_warp_id = local_id / WARP_SIZE;
//    const int lane_id = (WARP_SIZE - 1) & local_id;
//    int starting_x = (global_id / (warp_per_block * WARP_SIZE)) * warp_per_block;
//
//
//    // Prefetch
//    const int pos = d_cscColPtr[global_x_id];
//    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
//
//    if (local_id < warp_per_block) { s_graphInDegree[local_id] = 1; s_left_sum[local_id] = 0; }
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    // Consumer
//    int loads, loadd;
//    do {
//        // busy-wait
//    }
//    while ((loads = atomic_load_explicit((atomic_int*)&s_graphInDegree[local_warp_id],
//                                         memory_order_acquire, memory_scope_work_group)) !=
//           (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],
//                                         memory_order_acquire, memory_scope_device)) );
//
//    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];
//    xi = (d_b[global_x_id] - xi) * coef;
//
//    // Producer
//    const int start_ptr = d_cscColPtr[global_x_id]+1;
//    const int stop_ptr  = d_cscColPtr[global_x_id+1];
//    for (int j = start_ptr + lane_id; j < stop_ptr; j += WARP_SIZE) {
//        const int rowIdx = d_cscRowIdx[j];
//        const bool cond = (rowIdx < starting_x + warp_per_block);
//        if (cond) {
//            const int pos = rowIdx - starting_x;
//            if (sizeof(VALUE_TYPE) == 8)
//                atom_add_s_fp64(&s_left_sum[pos], xi * d_cscVal[j]);
//            else
//                atom_add_s_fp32(&s_left_sum[pos], xi * d_cscVal[j]);
//            mem_fence(CLK_LOCAL_MEM_FENCE);
//            atomic_fetch_add_explicit((atomic_int*)&s_graphInDegree[pos], 1,
//                                      memory_order_acquire, memory_scope_work_group);
//        }
//        else {
//            atom_add_d_fp(&d_left_sum[rowIdx], xi * d_cscVal[j]);
//            mem_fence(CLK_GLOBAL_MEM_FENCE);
//            atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
//                                       memory_order_acquire, memory_scope_device);
//        }
//    }
//
//    // Finish
//    if (!lane_id) d_x[global_x_id] = xi ;
}


