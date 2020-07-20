#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>


#include<iostream>
#include<sys/time.h>
//#include<stdlib.h>
//#include<stdio.h>
#include<cuda.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 100
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   32
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK  32
#endif

#ifndef MAX_BLOCK
#define MAX_BLOCK 65000
#endif
//
//#ifndef CPU_CU_NUM
//#define CPU_CU_NUM  4
//#endif

