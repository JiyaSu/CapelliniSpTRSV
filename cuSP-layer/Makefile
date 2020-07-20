
main: main.cu
	nvcc -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 main.cu -o main -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcusparse -lcudart
# 	nvcc -c -I/usr/local/cuda/include -gencode=arch=compute_61,code=compute_61 main.cu
# 	g++ -o main main.o -L/usr/local/cuda/lib64 -lcusparse -lcudart

clean:
	rm -f main