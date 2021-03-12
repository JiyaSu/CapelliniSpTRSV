#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif
#define WARP_SIZE 64

#define SUBSTITUTION_FORWARD 0
#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3

inline
void atom_add_d_fp32(volatile __global float *val,
                   float delta)
{
    union { float f; unsigned int i; } old;
    union { float f; unsigned int i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    }
    while (atomic_cmpxchg((volatile __global unsigned int *)val, old.i, new.i) != old.i);
}

inline
void atom_add_d_fp64(volatile __global double *val,
                   double delta)
{
    union { double f; ulong i; } old;
    union { double f; ulong i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    }
    while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);
}
inline
void atom_add_s_fp32(volatile __local float *val,
                   float delta)
{
    union { float f; unsigned int i; } old;
    union { float f; unsigned int i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    }
    while (atomic_cmpxchg((volatile __local unsigned int *)val, old.i, new.i) != old.i);
}

inline
void atom_add_d_fp(volatile __global VALUE_TYPE *address,
                   VALUE_TYPE val)
{
    if (sizeof(VALUE_TYPE) == 8)
        atom_add_d_fp64(address, val);
    else
        atom_add_d_fp32(address, val);
}

inline
void atom_add_s_fp64(volatile __local double *val,
                   double delta)
{
    union { double f; ulong i; } old;
    union { double f; ulong i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    }
    while (atom_cmpxchg((volatile __local ulong *)val, old.i, new.i) != old.i);
}

__kernel
void sptrsv_syncfree_opencl_analyser(__global const int      *d_cscRowIdx,
                                     const int                m,
                                     const int                nnz,
                                     __global int            *d_graphInDegree)
{
    const int global_id = get_global_id(0);
    if (global_id < nnz)
    {
        atomic_fetch_add_explicit((atomic_int*)&d_graphInDegree[d_cscRowIdx[global_id]], 1,
                                  memory_order_acq_rel, memory_scope_device);
    }
}

__kernel
void sptrsv_syncfree_opencl_executor(__global const int            *d_cscColPtr,
                                     __global const int            *d_cscRowIdx,
                                     __global const VALUE_TYPE     *d_cscVal,
                                     __global volatile int         *d_graphInDegree,
                                     __global volatile VALUE_TYPE  *d_left_sum,
                                     const int                      m,
                                     __global const VALUE_TYPE     *d_b,
                                     __global VALUE_TYPE           *d_x,
                                     __local volatile int          *s_graphInDegree,
                                     __local volatile VALUE_TYPE   *s_left_sum,
                                     const int                      warp_per_block)
{
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m) return;
    
   
    
    // Initialize
    const int local_warp_id = local_id / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & local_id;
    int starting_x = (global_id / (warp_per_block * WARP_SIZE)) * warp_per_block;
    
    
    // Prefetch
    const int pos = d_cscColPtr[global_x_id];
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];
    
    if (local_id < warp_per_block) { s_graphInDegree[local_id] = 1; s_left_sum[local_id] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Consumer
    int loads, loadd;
    do {
        // busy-wait
    }
    while ((loads = atomic_load_explicit((atomic_int*)&s_graphInDegree[local_warp_id],
                                         memory_order_acquire, memory_scope_work_group)) !=
           (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],
                                         memory_order_acquire, memory_scope_device)) );
    
    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;
    
    // Producer
    const int start_ptr = d_cscColPtr[global_x_id]+1;
    const int stop_ptr  = d_cscColPtr[global_x_id+1];
    for (int j = start_ptr + lane_id; j < stop_ptr; j += WARP_SIZE) {
        const int rowIdx = d_cscRowIdx[j];
        const bool cond = (rowIdx < starting_x + warp_per_block);
        if (cond) {
            const int pos = rowIdx - starting_x;
            if (sizeof(VALUE_TYPE) == 8)
                atom_add_s_fp64(&s_left_sum[pos], xi * d_cscVal[j]);
            else
                atom_add_s_fp32(&s_left_sum[pos], xi * d_cscVal[j]);
            mem_fence(CLK_LOCAL_MEM_FENCE);
            atomic_fetch_add_explicit((atomic_int*)&s_graphInDegree[pos], 1,
                                      memory_order_acquire, memory_scope_work_group);
        }
        else {
            atom_add_d_fp(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
                                       memory_order_acquire, memory_scope_device);
        }
    }
    
    // Finish
    if (!lane_id) d_x[global_x_id] = xi ;
}


