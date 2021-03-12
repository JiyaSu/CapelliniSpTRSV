#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 100
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   64
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK   4
#endif

#define SUBSTITUTION_FORWARD  0
#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3
