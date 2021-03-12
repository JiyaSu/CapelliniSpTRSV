#compilers
CC=g++

#GLOBAL_PARAMETERS
VALUE_TYPE = double

#includes
INCLUDES = -I/opt/rocm/opencl/include

#libs
OPENCL_LIBS = -L/opt//rocm/opencl/lib/x86_64 -lOpenCL
#OPENCL_LIBS = -framework OpenCL
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
LIBS = $(OPENCL_LIBS) $(CLANG_LIBS)

make:
	$(CC) -Wno-switch -Wno-write-strings -m64 -O3 main.cpp -o sptrsv $(INCLUDES) $(LIBS) -D VALUE_TYPE=$(VALUE_TYPE)
