
/* DPCT_ORIG #include "cuda_runtime.h"*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/* DPCT_ORIG #include "device_launch_parameters.h"*/
using namespace std;
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include<windows.h>
#include<fstream>
#include <ctime>
#include <time.h>

/* DPCT_ORIG cudaError_t mulwithCuda(int* m1, int* m2,int* result, int m, int n,
 * int k);*/
dpct::err0 mulwithCuda(int *m1, int *m2, int *result, int m, int n, int k);
int sdiv(int x, int y);
#define stepsize 4
#define blockW 4
/* DPCT_ORIG __global__ void mulKernel(int* m1, int* m2, int*result, int m,int
 * n,int k)*/
void mulKernel(int *m1, int *m2, int *result, int m, int n, int k,
               const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int col = blockDim.x * blockIdx.x + threadIdx.x;*/
    int col = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
/* DPCT_ORIG     int row = blockDim.y * blockIdx.y + threadIdx.y;*/
    int row = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
              item_ct1.get_local_id(1);
    int re = 0;
     
        for (int i = 0; i < n; i++) {
            re += m1[row * n + i] * m2[i * k + col];
        }
        int index = row * k + col;
        result[index] = re;
    
}
int main()
{
    int* m1;  //m*n
    int* m2;  //n*k
    int* result;
    // Add vectors in parallel.
    int m, n, k;
    m = 1600; n =1600; k = 1600;
    clock_t start, stop;
/* DPCT_ORIG     cudaMallocHost(&m1, m * n * sizeof(int));*/
    m1 = sycl::malloc_host<int>(m * n, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaMallocHost(&m2, k * n * sizeof(int));*/
    m2 = sycl::malloc_host<int>(k * n, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaMallocHost(&result, m * k * sizeof(int));*/
    result = sycl::malloc_host<int>(m * k, dpct::get_in_order_queue());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            m1[i * n + j] = i;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            m2[i * k + j] = i;
        }
    }
    int* m1_gpu, * m2_gpu, * result_gpu;


    clock_t start2 = clock();

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            result[i * k + j] = 0;
            for (int x = 0; x < n; x++) {
                result[i * k + j] += m1[i * n + x] * m2[x * k + j];
            }
        }
    }

    clock_t stop2 = clock();
    double elapsedTime = static_cast<double>(stop2 - start2) / CLOCKS_PER_SEC;

    // Print the result and execution time
    std::cout << "Matrix multiplication completed" << std::endl;
    std::cout << "CPU TIME: " << elapsedTime << " seconds" << std::endl;

    // Choose which GPU to run on, change this on a multi-GPU system.
/* DPCT_ORIG     cudaSetDevice(0);*/
    /*
    DPCT1093:4: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    dpct::select_device(0);
    // Allocate GPU buffers for three vectors (two input, one output)    .
/* DPCT_ORIG     cudaMalloc(&m1_gpu, m * n * sizeof(int));*/
    m1_gpu = sycl::malloc_device<int>(m * n, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaMalloc(&m2_gpu, k * n * sizeof(int));*/
    m2_gpu = sycl::malloc_device<int>(k * n, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaMalloc(&result_gpu, m * k * sizeof(int));*/
    result_gpu = sycl::malloc_device<int>(m * k, dpct::get_in_order_queue());

    // Copy input vectors from host memory to GPU buffers.
/* DPCT_ORIG     cudaMemcpy(m1_gpu, m1, m * n * sizeof(int),
 * cudaMemcpyHostToDevice);*/
    dpct::get_in_order_queue().memcpy(m1_gpu, m1, m * n * sizeof(int)).wait();
/* DPCT_ORIG     cudaMemcpy(m2_gpu, m2, n * k * sizeof(int),
 * cudaMemcpyHostToDevice);*/
    dpct::get_in_order_queue().memcpy(m2_gpu, m2, n * k * sizeof(int)).wait();
    // Launch a kernel on the GPU with one thread for each element.
    /*for (int blockW = 2; blockW <= 32; blockW *= 2) {
        
    }*/
    //按照维度定义的第一个x,第二个y
/* DPCT_ORIG     dim3 threadperblock(blockW, blockW);*/
    sycl::range<3> threadperblock(1, blockW, blockW);
/* DPCT_ORIG     dim3 numsblock((k + blockW - 1) / blockW, (m + blockW - 1) /
 * blockW);*/
    sycl::range<3> numsblock(1, (m + blockW - 1) / blockW,
                             (k + blockW - 1) / blockW);

    start = clock();

/* DPCT_ORIG     mulKernel<< <numsblock, threadperblock >> > (m1_gpu, m2_gpu,
 * result_gpu, m, n, k);*/
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(numsblock * threadperblock, threadperblock),
        [=](sycl::nd_item<3> item_ct1) {
            mulKernel(m1_gpu, m2_gpu, result_gpu, m, n, k, item_ct1);
        });
/* DPCT_ORIG     cudaDeviceSynchronize();*/
    dpct::get_current_device().queues_wait_and_throw();

    stop = clock();
    std::cout << "GPU TIME: " << static_cast<double>(stop - start) / CLOCKS_PER_SEC<< " seconds" << std::endl;
    // Copy output vector from GPU buffer to host memory.
/* DPCT_ORIG     cudaMemcpy(result, result_gpu, m * k * sizeof(int),
 * cudaMemcpyDeviceToHost);*/
    dpct::get_in_order_queue()
        .memcpy(result, result_gpu, m * k * sizeof(int))
        .wait();
/* DPCT_ORIG     cudaFree(m1_gpu);*/
    sycl::free(m1_gpu, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaFree(m2_gpu);*/
    sycl::free(m2_gpu, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaFree(result_gpu);*/
    sycl::free(result_gpu, dpct::get_in_order_queue());
    std::ofstream file("result.txt");
    if (file) {
        for (int i = 0; i < m*k; ++i) {
            file << result[i] << " ";
        }
        file.close();
    }
    else {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
/* DPCT_ORIG     cudaFreeHost(m1);*/
    sycl::free(m1, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaFreeHost(m2);*/
    sycl::free(m2, dpct::get_in_order_queue());
/* DPCT_ORIG     cudaFreeHost(result);*/
    sycl::free(result, dpct::get_in_order_queue());
    return 0;
}
int sdiv(int x, int y) {
    if (y == 0) {
        return 0;
    }

    int result = (x + y - 1) / y;
    return result;
}

// Helper function for using CUDA to add vectors in parallel.
