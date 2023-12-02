#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>

// 定义 CUDA 核函数
void convolutionKernel(const int* input, const int* kernel, int* output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int outputWidth, int outputHeight,
                       const sycl::nd_item<3> &item_ct1)
{
    int x = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    int y = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);

    if (x < outputWidth && y < outputHeight)
    {
        int sum = 0;
        for (int ky = 0; ky < kernelHeight; ky++)
        {
            for (int kx = 0; kx < kernelWidth; kx++)
            {
                int inputX = x + kx;
                int inputY = y + ky;
                sum += input[inputY * inputWidth + inputX] * kernel[ky * kernelWidth + kx];
            }
        }
        output[y * outputWidth + x] = sum;
    }
}

// 定义卷积函数
void convolution(const std::vector<int>& input, const std::vector<int>& kernel, std::vector<int>& output, int inputWidth, int inputHeight, int kernelWidth, int kernelHeight)
{
    sycl::device dev_ct1;
    sycl::queue q_ct1(dev_ct1,
                      sycl::property_list{sycl::property::queue::in_order()});
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;
    int inputSize = inputWidth * inputHeight * sizeof(int);
    int kernelSize = kernelWidth * kernelHeight * sizeof(int);
    int outputSize = outputWidth * outputHeight * sizeof(int);

    // 在设备上分配内存
    int* d_input;
    int* d_kernel;
    int* d_output;
    d_input = (int *)sycl::malloc_device(inputSize, q_ct1);
    d_kernel = (int *)sycl::malloc_device(kernelSize, q_ct1);
    d_output = (int *)sycl::malloc_device(outputSize, q_ct1);

    // 将输入数据复制到设备内存
    q_ct1.memcpy(d_input, input.data(), inputSize).wait();
    q_ct1.memcpy(d_kernel, kernel.data(), kernelSize).wait();

    // 定义线程块和网格的大小
    sycl::range<3> blockSize(1, 16, 16);
    sycl::range<3> gridSize(1, (outputHeight + blockSize[1] - 1) / blockSize[1],
                            (outputWidth + blockSize[2] - 1) / blockSize[2]);

    // 调用 CUDA 核函数进行卷积运算
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                       [=](sycl::nd_item<3> item_ct1) {
                           convolutionKernel(
                               d_input, d_kernel, d_output, inputWidth,
                               inputHeight, kernelWidth, kernelHeight,
                               outputWidth, outputHeight, item_ct1);
                       });

    // 将计算结果复制回主机内存
    q_ct1.memcpy(output.data(), d_output, outputSize).wait();

    // 释放设备内存
    sycl::free(d_input, q_ct1);
    sycl::free(d_kernel, q_ct1);
    sycl::free(d_output, q_ct1);
}

void test()
{
    // 定义输入图像和卷积核
    std::vector<int> input = { 1, 2, 3, 4, 5,
                               6, 7, 8, 9, 10,
                               11, 12, 13, 14, 15,
                               16, 17, 18, 19, 20,
                               21, 22, 23, 24, 25 };
    std::vector<int> kernel = { 1, 5, -1,
                                2, 0, -2,
                                1, 0, -1 };

    int inputWidth = 5;
    int inputHeight = 5;
    int kernelWidth = 3;
    int kernelHeight = 3;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    // 创建输出图像
    std::vector<int> output(outputWidth * outputHeight);

    // 进行卷积运算
    convolution(input, kernel, output, inputWidth, inputHeight, kernelWidth, kernelHeight);

    // 打印输出图像
    for (int y = 0; y < outputHeight; y++)
    {
        for (int x = 0; x < outputWidth; x++)
        {
            std::cout << output[y * outputWidth + x] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    test();

    return 0;
}