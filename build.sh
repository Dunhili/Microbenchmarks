nvcc -O3 -Xopencc="-LIST:source=on -O3" -arch=sm_21 -o throughput cuda_integer_benchmark.cu
