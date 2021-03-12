#ifndef _UTILS_
#define _UTILS_

#include "common.h"

// print 1D array
template<typename T>
void print_1darray(T *input, int length)
{
    for (int i = 0; i < length; i++)
        printf("%i, ", input[i]);
    printf("/n");
}

/*struct assembly_timer {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }

    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};*/

template<typename T>
void swap(T *a , T *b)
{
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key-value pair (child function)
template<typename iT, typename vT>
int partition(iT *key, vT *val, int length, int pivot_index)
{
    int i  = 0 ;
    int small_length = pivot_index;

    iT pivot = key[pivot_index];
    swap<iT>(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap<vT>(&val[pivot_index], &val[pivot_index + (length - 1)]);

    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap<iT>(&key[pivot_index+i],  &key[small_length]);
            swap<vT>(&val[pivot_index+i],&val[small_length]);
            small_length++;
        }
    }

    swap<iT>(&key[pivot_index + length - 1],  &key[small_length]);
    swap<vT>(&val[pivot_index + length - 1],&val[small_length]);

    return small_length;
}

// quick sort key-value pair (main function)
template<typename iT, typename vT>
void quick_sort_key_val_pair(iT *key, vT *val, int length)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition<iT, vT>(key, val, length, 0) ;
    quick_sort_key_val_pair<iT, vT>(key, val, small_length);
    quick_sort_key_val_pair<iT, vT>(&key[small_length + 1], &val[small_length + 1], length - small_length - 1);
}
/*
template<typename iT>
void move_block(iT* first,
                iT* last,
                iT* result)
{
    //memcpy(result, first, sizeof(iT) * (last - first));
    while (first != last)
    {
        *result = *first;
        ++result;
        ++first;
    }
}

template<typename iT, typename vT>
void serial_merge(iT* key_left_start,
                  iT* key_left_end,
                  iT* key_right_start,
                  iT* key_right_end,
                  iT* key_output,
                  vT* val_left_start,
                  vT* val_left_end,
                  vT* val_right_start,
                  vT* val_right_end,
                  vT* val_output)
{
    while(key_left_start != key_left_end && key_right_start != key_right_end)
    {
        bool which = *key_right_start < *key_left_start;
        //*key_output++ = std::move(which ? *key_right_start++ : *key_left_start++);
        *key_output++ = which ? *key_right_start++ : *key_left_start++;
        *val_output++ = which ? *val_right_start++ : *val_left_start++;
    }

    //std::move( key_left_start, key_left_end, key_output );
    move_block<iT>(key_left_start, key_left_end, key_output);
    move_block<vT>(val_left_start, val_left_end, val_output);

    //std::move( key_right_start, key_right_end, key_output );
    move_block<iT>(key_right_start, key_right_end, key_output);
    move_block<vT>(val_right_start, val_right_end, val_output);
}

// merge sequences [key_left_start,key_left_end) and [key_right_start,key_right_end)
// to output [key_output, key_output+(key_left_end-key_left_start)+(key_right_end-key_right_start))
template<typename iT, typename vT>
void parallel_merge(iT* key_left_start,
                    iT* key_left_end,
                    iT* key_right_start,
                    iT* key_right_end,
                    iT* key_output,
                    vT* val_left_start,
                    vT* val_left_end,
                    vT* val_right_start,
                    vT* val_right_end,
                    vT* val_output)
{
    const size_t MERGE_CUT_OFF = 2000;

    if( key_left_end - key_left_start + key_right_end - key_right_start <= MERGE_CUT_OFF)
    {
        serial_merge<iT, vT>(key_left_start, key_left_end, key_right_start, key_right_end, key_output,
                             val_left_start, val_left_end, val_right_start, val_right_end, val_output);
    }
    else
    {
        iT *key_left_middle, *key_right_middle;
        vT *val_left_middle, *val_right_middle;

        if(key_left_end - key_left_start < key_right_end - key_right_start)
        {
            key_right_middle = key_right_start + (key_right_end - key_right_start) / 2;
            val_right_middle = val_right_start + (val_right_end - val_right_start) / 2;

            key_left_middle = std::upper_bound(key_left_start, key_left_end, *key_right_middle);
            val_left_middle = val_left_start + (key_left_middle - key_left_start);
        }
        else
        {
            key_left_middle = key_left_start + (key_left_end - key_left_start) / 2;
            val_left_middle = val_left_start + (val_left_end - val_left_start) / 2;

            key_right_middle = std::lower_bound(key_right_start, key_right_end, *key_left_middle);
            val_right_middle = val_right_start + (key_right_middle - key_right_start);
        }

        iT* key_output_middle = key_output + (key_left_middle - key_left_start) + (key_right_middle - key_right_start);
        iT* val_output_middle = val_output + (val_left_middle - val_left_start) + (val_right_middle - val_right_start);

#pragma omp task
        parallel_merge<iT, vT>(key_left_start,  key_left_middle, key_right_start,  key_right_middle, key_output,
                               val_left_start,  val_left_middle, val_right_start,  val_right_middle, val_output);
        parallel_merge<iT, vT>(key_left_middle, key_left_end,    key_right_middle, key_right_end,    key_output_middle,
                               val_left_middle, val_left_end,    val_right_middle, val_right_end,    val_output_middle);
#pragma omp taskwait
    }
}

// sorts [key_start,key_end).
// key_temp[0:key_end-key_start) is temporary buffer supplied by caller.
// result is in [key_start,key_end) if inplace==true,
// otherwise in key_temp[0:key_end-key_start).
template<typename iT, typename vT>
void parallel_merge_sort(iT* key_start,
                         iT* key_end,
                         iT* key_temp,
                         vT* val_start,
                         vT* val_end,
                         vT* val_temp,
                         bool inplace)
{
    const size_t SORT_CUT_OFF = 500;

    if(key_end - key_start <= SORT_CUT_OFF)
    {
        //std::stable_sort(key_start, key_end);
        int list_length = key_end - key_start;
        quick_sort_key_val_pair(key_start, val_start, list_length);

        if(!inplace)
        {
            //std::move( key_start, key_end, key_temp );
            move_block<iT>(key_start, key_end, key_temp);
            move_block<vT>(val_start, val_end, val_temp);
        }
    }
    else
    {
        iT* key_middle = key_start + (key_end - key_start) / 2;
        vT* val_middle = val_start + (val_end - val_start) / 2;
        iT* key_temp_middel = key_temp + (key_middle - key_start);
        vT* val_temp_middel = val_temp + (val_middle - val_start);
        iT* key_temp_end = key_temp + (key_end - key_start);
        vT* val_temp_end = val_temp + (val_end - val_start);

#pragma omp task
        parallel_merge_sort<iT, vT>(key_start,  key_middle, key_temp,
                                    val_start,  val_middle, val_temp,
                                    !inplace);
        parallel_merge_sort<iT, vT>(key_middle, key_end,    key_temp_middel,
                                    val_middle, val_end,    val_temp_middel,
                                    !inplace);
#pragma omp taskwait
        if(inplace)
            parallel_merge<iT, vT>(key_temp, key_temp_middel, key_temp_middel, key_temp_end, key_start,
                                   val_temp, val_temp_middel, val_temp_middel, val_temp_end, val_start);
        else
            parallel_merge<iT, vT>(key_start, key_middle, key_middle, key_end, key_temp,
                                   val_start, val_middle, val_middle, val_end, val_temp);
   }
}

// OpenMP tasks do not run in parallel unless launched inside a thread team.
// This outer wrapper shows how to create the thread team and run the top-level call.
template<typename iT, typename vT>
void do_parallel_merge_sort(iT* key_start,
                            iT* key_end,
                            iT* key_temp,
                            vT* val_start,
                            vT* val_end,
                            vT* val_temp,
                            bool inplace)
{
    // Create a thread team.
#pragma omp parallel
    // Make only one thread do the top-level call.
    // Other threads in team pick up spawned tasks.
#pragma omp single
    {
        parallel_merge_sort<iT, vT>(key_start, key_end, key_temp,
                                    val_start, val_end, val_temp,
                                    inplace);
    }
}

// merge sort key-value pair (main function)
template<typename iT, typename vT>
void omp_merge_sort_key_val_pair(iT *key, vT *val, int length)
{
    //quick_sort_key_val_pair<iT, vT>(key, val, length);

    if(length == 0 || length == 1)
        return;

    // allocate temp space for out-of-place merge sort
    iT *key_temp = (iT *)malloc(length * sizeof(iT));
    vT *val_temp = (vT *)malloc(length * sizeof(vT));

    bool inplace = true;
    do_parallel_merge_sort<iT, vT>(&key[0], &key[length], key_temp,
                                   &val[0], &val[length], val_temp,
                                   inplace);

    // free temp space
    free(key_temp);
    free(val_temp);
}*/

// in-place exclusive scan
template<typename T>
void exclusive_scan(T *input, int length)
{
    if(length == 0 || length == 1)
        return;

    T old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

// segmented sum
template<typename vT, typename bT>
void segmented_sum(vT *input, bT *bit_flag, int length)
{
    if(length == 0 || length == 1)
        return;

    for (int i = 0; i < length; i++)
    {
        if (bit_flag[i])
        {
            int j = i + 1;
            while (!bit_flag[j] && j < length)
            {
                input[i] += input[j];
                j++;
            }
        }
    }
}

// reduce sum
template<typename T>
T reduce_sum(T *input, int length)
{
    if(length == 0)
        return 0;

    T sum = 0;

    for (int i = 0; i < length; i++)
    {
        sum += input[i];
    }

    return sum;
}

#endif
