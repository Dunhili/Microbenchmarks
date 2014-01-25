/*
 * Author: Brian Bowden
 * Date: 1/25/14
 *
 * cuda_integer_benchmark.cu
 * 
 * Microbenchmarks designed to test find the throughput of ints, u_ints, floats, and doubles.
 * The kernels for each instruction are designed so that the compiler doesn't optimize the
 * instructions out and each kernel will perform each instruction as we want. The times are
 * normalized using the clock speed of the GPU and the number of instructions to get an 
 * instruction/cycle result. 
 */

#include <stdlib.h>
#include <stdio.h>
#include "repeat.h"

#define REPEAT(iters)	repeat ## iters

void print_results(double average_time, int iterations);
void getThroughput(int benchmark, int iterations);
void gpu_init();
int gcd(int a, int b);

enum Data_Types {
    INT, UINT, FLOAT, DOUBLE
};

enum Instructions {
    Add, Sub, Mul, Div, MAD, VAdd, AND, OR, XOR, SHL, SHR, LRot, RRot
};

// change two lines below if you want to test Integers or Unsigned Integers
typedef int TYPE;
#define DATATYPE (INT)

// constants
const int number_runs = 25;
const int instructions_per_repeat = 4;

// updated in the gpu_init function
float clock_speed;
int number_multi_processors;
int number_blocks;
int number_threads;
int max_threads_per_mp;
int block_size;

// host arrays
TYPE* host_A;
TYPE* host_B;
TYPE* host_C;
TYPE* host_D;

// device arrays
TYPE* device_A;
TYPE* device_B;
TYPE* device_C;
TYPE* device_D;
	
cudaEvent_t start, stop;

__global__ void kernelAdd(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val += b_val;
               b_val += c_val;
               c_val += d_val;
               d_val += a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelSub(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    TYPE e_val = E[i];
    repeat4096(a_val -= b_val;
               b_val -= c_val;
               c_val -= d_val;
               d_val -= e_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelMul(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val *= b_val;
               b_val *= c_val;
               c_val *= d_val;
               d_val *= a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelDiv(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat256(a_val /= b_val;
              b_val /= c_val;
              c_val /= d_val;
              d_val /= a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelMAD(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val *= b_val; a_val += b_val;
               b_val *= c_val; b_val += c_val;
               c_val *= d_val; c_val += d_val;
               d_val *= a_val; d_val += a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelVectorAdd(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat2048(a_val += b_val + c_val;
               b_val += c_val + d_val;
               c_val += d_val + a_val;
               d_val += a_val + b_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelRemainder(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];	
    repeat256(a_val %= b_val;
              b_val %= c_val;
              c_val %= d_val;
              d_val %= a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelAND(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val = b_val & c_val;
               b_val = c_val & d_val;
               c_val = d_val & a_val;
               d_val = a_val & b_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelOR(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val = b_val | c_val;
               b_val = c_val | d_val;
               c_val = d_val | a_val;
               d_val = a_val | b_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelXOR(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val = b_val ^ c_val;
               b_val = c_val ^ d_val;
               c_val = d_val ^ a_val;
               d_val = a_val ^ b_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelShl(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val <<= b_val;
               b_val <<= c_val;
               c_val <<= d_val;
               d_val <<= a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelShr(TYPE* A, TYPE* B, TYPE* C, TYPE* D) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat4096(a_val >>= b_val;
               b_val >>= c_val;
               c_val >>= d_val;
               d_val >>= a_val;);
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelLeftRotate(TYPE* A, TYPE* B, TYPE* C, TYPE* D, int shift) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat1024(a_val = (b_val << shift) | (b_val >> (32 - shift)); 
               b_val = (c_val << shift) | (c_val >> (32 - shift)); 
               c_val = (d_val << shift) | (d_val >> (32 - shift)); 
               d_val = (a_val << shift) | (a_val >> (32 - shift)););
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

__global__ void kernelRightRotate(TYPE* A, TYPE* B, TYPE* C, TYPE* D, int shift) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    TYPE a_val = A[i];
    TYPE b_val = B[i];
    TYPE c_val = C[i];
    TYPE d_val = D[i];
    repeat1024(a_val = (b_val >> shift) | (b_val << (32 - shift)); 
               b_val = (c_val >> shift) | (c_val << (32 - shift)); 
               c_val = (d_val >> shift) | (d_val << (32 - shift)); 
               d_val = (a_val >> shift) | (a_val << (32 - shift)););
    A[i] = a_val;
    B[i] = b_val;
    C[i] = c_val;
    D[i] = d_val;
}

/*
 * Prints out the results for the current throughput test. 
 */
void print_results(double average_time, int iterations) {
    int number_instructions = max_threads_per_mp * number_multi_processors * iterations * instructions_per_repeat;
    long number_cycles = (long) ((average_time / 1000) * clock_speed);
    double throughput = ((double) number_instructions) / ((double) number_cycles);
    printf("%0.3g\n", throughput);
}

/*
 * Prints out and calls the appropriate throughput test.
 */
void getThroughput(Instructions instr, int iterations) {
    switch (instr) {
        case Add:  printf("Addition:        "); break;
        case Sub:  printf("Subtraction:     "); break;
        case Mul:  printf("Multiplication:  "); break;
        case Div:  printf("Division:        "); break;
        case MAD:  printf("Multiply-Add:    "); break;
        case VAdd: printf("Vector-Addition: "); break;
        case Rem:  printf("Remainder:       "); break;
        case AND:  printf("AND:             "); break;
        case OR:   printf("OR:              "); break;
        case XOR:  printf("XOR:             "); break;
        case SHL:  printf("Shift-Left:      "); break;
        case SHR:  printf("Shift-Right:     "); break;
        case LRot: printf("Left-Rotate:     "); break;
        case RRot: printf("Right-Rotate:    "); break;
    }

    double average_time = 0.0;
    float time_elapsed;

    //int shift = 8;
    for (int j = 0; j < number_runs; j++) {
        cudaEventRecord(start, 0);
        switch (instr) {
            case Add:  kernelAdd<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;	
            case Sub:  kernelSub<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;	
            case Mul:  kernelMul<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;	
            case Div:  kernelDiv<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;	
            case MAD:  kernelMAD<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;
            case VAdd: kernelVectorAdd<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);            break;	
            case Rem:  kernelRemainder<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);            break;
            case AND:  kernelAND<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;
            case OR:   kernelOR<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                   break;
            case XOR:  kernelXOR<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;
            case SHL:  kernelShl<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;
            case SHR:  kernelShr<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D);                  break;
            case LRot: kernelLeftRotate<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D, shift);    break;
            case RRot: kernelRightRotate<<<number_blocks, number_threads>>>(device_A, device_B, device_C, device_D, shift);   break;
        }
	
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed, start, stop);
        average_time += time_elapsed;
    }

    print_results(average_time / number_runs, iterations);
}

/**
 * Initializes the global variables by calling the cuda
 */
void gpu_init() {
    cudaDeviceProp device_prop;
    int device_count;

    cudaGetDeviceCount(&device_count);
    if (device_count != 1) {
        printf("Only want to test a single GPU, exiting...\n");
        exit(EXIT_FAILURE);
    }

    if (cudaGetDeviceProperties(&device_prop, 0) != cudaSuccess) {
        printf("Problem getting properties for device, exiting...\n");
        exit(EXIT_FAILURE);
    } 

    number_threads = device_prop.maxThreadsPerBlock;
    number_multi_processors = device_prop.multiProcessorCount;
    max_threads_per_mp = device_prop.maxThreadsPerMultiProcessor;
    
    block_size = (max_threads_per_mp / gcd(max_threads_per_mp, number_threads));
    number_threads = max_threads_per_mp / block_size;
    number_blocks = number_multi_processors * block_size;
    clock_speed = device_prop.memoryClockRate * 1000;
}

int gcd(int a, int b) {
    if (a == 0)
	    return b;
    return gcd (b % a, a);
}

int main(int argc, char **argv) {
    gpu_init();
    const int N = max_threads_per_mp * number_multi_processors;
    size_t array_size = N * sizeof(TYPE);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host arrays
    host_A = (TYPE *) malloc(array_size);
    host_B = (TYPE *) malloc(array_size);
    host_C = (TYPE *) malloc(array_size);
    host_D = (TYPE *) malloc(array_size);

    if (host_A == NULL || host_B == NULL || host_C == NULL || host_D == NULL) {
        printf("Failed allocating array(s), exiting...\n");
        exit(EXIT_FAILURE);
    }

    //Initilize arrays
    for (int i = 0; i < N; i++) {
        host_A[i] = i * 10000;
        host_B[i] = i * 1000;
        host_C[i] = i * 100;
        host_D[i] = i * 10;
    }

    // Allocate device arrays
    cudaMalloc((void**) &device_A, array_size);
    cudaMalloc((void**) &device_B, array_size);
    cudaMalloc((void**) &device_C, array_size);
    cudaMalloc((void**) &device_D, array_size);

    // Copy ints from host to device arrays
    cudaMemcpy(device_A, host_A, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_D, host_D, array_size, cudaMemcpyHostToDevice);

    switch(DATATYPE) {
        case INT:    printf("Integer\n");          break;
        case UINT:   printf("Unsigned-Integer\n"); break;
        case FLOAT:  printf("Float\n");            break;
        case DOUBLE: printf("Double\n");           break;
    }

    getThroughput(Add,  4096);
    getThroughput(Sub,  4096);
    getThroughput(Mul,  4096);
    getThroughput(Div,   256);
    getThroughput(MAD,  4096);
    getThroughput(VAdd, 2048);
    getThroughput(Rem,   256);
    getThroughput(AND,  4096);
    getThroughput(OR,   4096);
    getThroughput(XOR,  4096);
    getThroughput(SHL,  4096);
    getThroughput(SHR,  4096);
    getThroughput(LRot, 1024);
    getThroughput(RRot, 1024);

    // Free arrays from memory
    free(host_A);
    free(host_B);
    free(host_C);
    free(host_D);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    cudaFree(device_D);

    return EXIT_SUCCESS;
}