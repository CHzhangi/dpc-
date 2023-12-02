#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
using namespace std;
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include<windows.h>
#include<fstream>
#include<ctime>
#include<vector>
#include<algorithm>
void mergesortK(int* a, int* temp, int sortedsize, int N,
                const sycl::nd_item<3> &item_ct1)
{
    // int id = blockIdx.x * blockDim.x + threadIdx.x;

    int blockid = item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                      item_ct1.get_group_range(1) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                  item_ct1.get_group(2);
    int id = blockid * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    unsigned long index1, index2, endIndex1, endIndex2, targetIndex;
    index1 = id * 2 * sortedsize;
    endIndex1 = index1 + sortedsize;
    index2 = endIndex1;
    endIndex2 = index2 + sortedsize;
    targetIndex = id * 2 * sortedsize;

    if (index1 >= N) return;

    if (endIndex1 > N)
    {
        endIndex1 = N;
        index2 = endIndex2 = N;
    }
    if (index2 > N)
    {
        index2 = endIndex2 = N;
    }
    if (endIndex2 > N)
        endIndex2 = N;



    int done = 0;
    while (!done)
    {
        if ((index1 == endIndex1) && (index2 < endIndex2))
            temp[targetIndex++] = a[index2++];
        else if ((index2 == endIndex2) && (index1 < endIndex1))
            temp[targetIndex++] = a[index1++];
        else if (a[index1] < a[index2])
            temp[targetIndex++] = a[index1++];
        else
            temp[targetIndex++] = a[index2++];

        if ((index1 == endIndex1) && (index2 == endIndex2))
            done = 1;
    }
}
int mergesort(int* data, int N, float& cost_time)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    int* dev_a, * dev_temp;

    dev_a = sycl::malloc_device<int>(N, q_ct1);
    dev_temp = sycl::malloc_device<int>(N, q_ct1);
   q_ct1.memcpy(dev_a, data, sizeof(int) * N).wait();
    int blocks = 512;
    sycl::range<3> grids(1, 1, 128);
    float t0 = GetTickCount();
    int sortedsize = 1;
    while (sortedsize < N)
    {
        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        q_ct1.parallel_for(
            sycl::nd_range<3>(grids * sycl::range<3>(1, 1, blocks),
                              sycl::range<3>(1, 1, blocks)),
            [=](sycl::nd_item<3> item_ct1) {
                mergesortK(dev_a, dev_temp, sortedsize, N, item_ct1);
            });
        q_ct1.memcpy(dev_a, dev_temp, N * sizeof(int));
        sortedsize *= 2;
    }

    q_ct1.memcpy(data, dev_a, N * sizeof(int)).wait();
    
    sycl::free(dev_a, q_ct1);
   sycl::free(dev_temp, q_ct1);

    dev_ct1.queues_wait_and_throw();
    cost_time = GetTickCount() - t0;
    return 0;
}
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // 创建临时数组存储左右两个子数组
    std::vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    // 合并左右两个子数组到原数组
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // 将剩余的元素复制回原数组
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// 归并排序
void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // 分割数组并递归排序左右子数组
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // 合并排序后的左右子数组
        merge(arr, left, mid, right);
    }
}

int main(int argc, char* argv[])
{
    int N = 102400;
    int* data = new int[N];
    std::vector<int> data_vec;
    for (int k = 0; k < N; k++)
    {
        data[k] = rand() % 4096;
        data_vec.push_back(data[k]);
        //std::cout << data[k] << ",";
    }
    std::cout << std::endl;

    //float t0 = GetTickCount();
    float cost_gpu;
    mergesort(data, N, cost_gpu);
    //float t1 = GetTickCount();

    float tt0 = GetTickCount();
    mergeSort(data_vec, 0, data_vec.size() - 1);
    float tt1 = GetTickCount();

    int flag = 0;
    for (int k = 0; k < N; k++)
    {
        if (data[k] == data_vec[k])
        {
            flag++;
        }
    }
    std::cout << std::endl;
    std::cout << "check result (" << flag << "," << N << ") = " << (flag == N) << std::endl;

    std::cout << "cpu cost " << cost_gpu << "ms" << std::endl;
    std::cout << "gpu cost " << tt1 - tt0 << "ms" << std::endl;
    return 0;
}