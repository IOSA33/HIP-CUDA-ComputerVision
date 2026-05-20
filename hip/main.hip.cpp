#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image_write.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <iterator>
#include <hip/hip_runtime.h>


//
//  Declrations
//
void HandVision(std::vector<unsigned char>& vec, std::vector<unsigned char>& mask);
double checkResults(const std::vector<unsigned char>& __mask_fromGPU, const std::vector<unsigned char>& __mask_for_CPU_test);

//
// Predefined photo size
//
constexpr int g_Width { 3072 };
constexpr int g_Height { 4096 };

__global__ void DrawRectangle(unsigned char* mask, int width, int height, unsigned long long sum_00, unsigned long long sum_10, unsigned long long sum_01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sum_00 == 0) return;
    unsigned long long x_coord { sum_10 / sum_00 };
    unsigned long long y_coord { sum_01 / sum_00 };

    int r { 1100 };
    int d { 1350 };

    if (idx <= 2 * r) {
        int nx = x_coord - r + idx;

        if (nx >= 0 && nx < width) {
            int ny1 = y_coord - d;
            int ny2 = y_coord + d;
            
            if (ny1 >= 0 && ny1 < height) {
                mask[ny1 * width + nx] = 255;
            }
            if (ny2 >= 0 && ny2 < height) {
                mask[ny2 * width + nx] = 255;
            }
        }
    }

    if (idx <= 2 * d) {
        int ny = y_coord - d + idx;

        if (ny >= 0 && ny < height) {
            int nx1 = x_coord - r;
            int nx2 = x_coord + r;

            if (nx1 >= 0 && nx1 < width) {
                mask[ny * width + nx1] = 255;
            }
            if (nx2 >= 0 && nx2 < width) {
                mask[ny * width + nx2] = 255;
            }
        }
    }
}

//
// Doing hand vision work with kernels
//
__global__ void HandVisionGPU(unsigned char* vec, unsigned char* mask, int width, int height, unsigned long long* sum_00, unsigned long long* sum_10, unsigned long long* sum_01) {
    // Getting thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Size of image
    int Y_size    = width * height;

    // The result may differ from one image to another, color skin
    unsigned char colR { 143 };
    unsigned char colG { 103 };
    unsigned char colB { 80 };
    unsigned char tolerance { 29 };

    // Shared memory for every block
    __shared__ unsigned long long s_sum_00;
    __shared__ unsigned long long s_sum_10;
    __shared__ unsigned long long s_sum_01;

    // Zeroing values
    if (threadIdx.x == 0) {
        s_sum_00 = 0;
        s_sum_10 = 0;
        s_sum_01 = 0;
    }
    // Synchronize that every thread has same correct shared memory
    __syncthreads();

    if (idx < Y_size) {
        int y = idx / width;
        int x = idx % width;
        
        // brightness index
        int Y = vec[idx];
        // Calculating index in the vector
        int uv_row = y / 2;
        int uv_col = x / 2;
        int uv_index = Y_size + (uv_row * width) + (uv_col * 2);
        // UV for the pixel (i,j)
        int U   = vec[uv_index];
        int V   = vec[uv_index + 1];

        // Doing calculations for correct values
        int C = Y - 16;
        int D = U - 128;
        int E = V - 128;

        // Getting RGB value
        int R = (298 * C           + 409 * E + 128) >> 8;
        int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
        int B = (298 * C + 516 * D           + 128) >> 8;
        
        // Validating the color
        if (R < 0)   R = 0; 
        if (R > 255) R = 255;
        if (G < 0)   G = 0; 
        if (G > 255) G = 255;
        if (B < 0)   B = 0;
        if (B > 255) B = 255;

        // If pixel is skin color
        if (abs(R - colR) <= tolerance && abs(G - colG) <= tolerance && abs(B - colB) <= tolerance) {
            // Putting white color to the pixel in the new mask vector
            mask[idx] = 255;
            
            // Doing atomic add in to shared memory
            atomicAdd(&s_sum_00, 1);
            atomicAdd(&s_sum_10, x);
            atomicAdd(&s_sum_01, y);

        } else {
            // Putting black color to the pixel in the mask vector
            mask[idx] =  0;
        }

        // Waiting that every thread is finnished
        __syncthreads();

        // Adding then with first thread in the block, shared memory values to the global memory
        if (threadIdx.x == 0) {
            atomicAdd(sum_00, s_sum_00);
            atomicAdd(sum_10, s_sum_10);
            atomicAdd(sum_01, s_sum_01);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "usage: ./app.exe <path_to_file> <test?_int_value_1_or_0>";
        return 1;
    }

    //
    // HIP error type for function checks
    //
    hipError_t err;
    
    //
    // Initializing variables for sentroid
    //
    unsigned long long host_sum_00 { 0 };
    unsigned long long host_sum_10 { 0 };
    unsigned long long host_sum_01 { 0 };

    //
    // Opening file and read raw bytes from it
    // For example path can be like this - "../photos/output.nv12" 
    //
    std::ifstream input( argv[1], std::ios::binary | std::ios::ate );
    if (!input.is_open()) {
        std::cout << "Cant open a file!\n";
        return 1;
    }

    //
    // Reserving merory for vector
    //
    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (size <= 0) {
        std::cout << "File is empty!" << std::endl;
        return 1;
    }

    //
    // Copying all data from file to the vector buffer
    //
    std::vector<unsigned char> __buffer(size);
    if (!input.read(reinterpret_cast<char*>(__buffer.data()), size)) {
        std::cout << "unable to copy data to vector!" << std::endl;
        return 1;
    }

    //
    // Reserving mask vector for output result
    //
    std::vector<unsigned char> __mask(g_Width * g_Height);
    std::vector<unsigned char> __mask_for_CPU_test(g_Width * g_Height);
    input.close();

    //
    // Checking that GPU is available
    //
    int deviceCount = 0;
    err = hipGetDeviceCount(&deviceCount);
    if ( err != hipSuccess ) {
        std::cerr << "Error getting a device count." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    //
    // Allocating space in global memory, and also setting with the value
    //
    unsigned char* deviceBuffer;
    unsigned char* deviceMask;
    unsigned long long* device_sum_00;
    unsigned long long* device_sum_10;
    unsigned long long* device_sum_01;

    err = hipMalloc(&deviceBuffer, size);
    if ( err != hipSuccess ) {
        std::cerr << "Failed to allocate memory." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc(&deviceMask, g_Width * g_Height * sizeof(unsigned char));
    if ( err != hipSuccess ) {
        std::cerr << "Failed to allocate memory." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc(&device_sum_00, sizeof(unsigned long long));
    if ( err != hipSuccess ) {
        std::cerr << "Failed to allocate memory." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc(&device_sum_10, sizeof(unsigned long long));
    if ( err != hipSuccess ) {
        std::cerr << "Failed to allocate memory." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMalloc(&device_sum_01, sizeof(unsigned long long));
    if ( err != hipSuccess ) {
        std::cerr << "Failed to allocate memory." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMemset(device_sum_00, 0, sizeof(unsigned long long));
    if ( err != hipSuccess ) {
        std::cerr << "Failed to SET memory value." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMemset(device_sum_10, 0, sizeof(unsigned long long));
    if ( err != hipSuccess ) {
        std::cerr << "Failed to SET memory value." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    err = hipMemset(device_sum_01, 0, sizeof(unsigned long long));
        if ( err != hipSuccess ) {
        std::cerr << "Failed to SET memory value." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    //
    // Transfering raw bytes __buffer to device memory, we dont transfer __mask because
    // its empty vector and we only gonna store it there the data
    //
    err = hipMemcpy(deviceBuffer, __buffer.data(), size, hipMemcpyHostToDevice);
    if ( err != hipSuccess ) {
        std::cerr << "Failed to copy memory to device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }

    //
    // Configuring blocks and threads
    //
    const dim3 numberOfBlocks((g_Width * g_Height - 1) / 256 + 1);
    const dim3 threadsPerBlock(256);

    //
    // Starting timer
    //
    const auto start { std::chrono::high_resolution_clock::now() };

    //
    // Kernel Calls
    //    
    HandVisionGPU<<<numberOfBlocks, threadsPerBlock>>>(deviceBuffer, deviceMask, g_Width, g_Height, device_sum_00, device_sum_10, device_sum_01);
    err = hipGetLastError();
    if ( err != hipSuccess ) {
        std::cerr << "Failed to invoke the kernel." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }
    //
    // Getting the result from the GPU to Host
    //
    err = hipMemcpy(&host_sum_00, device_sum_00, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }
    err = hipMemcpy(&host_sum_01, device_sum_01, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }
    err = hipMemcpy(&host_sum_10, device_sum_10, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }
    err = hipMemcpy(__mask.data(), deviceMask, g_Width * g_Height * sizeof(unsigned char), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }

    //
    // Calling DrawRectangle second Kernel
    //
    DrawRectangle<<<numberOfBlocks, threadsPerBlock>>>(deviceMask, g_Width, g_Height, host_sum_00, host_sum_10, host_sum_01);
    err = hipGetLastError();
    if ( err != hipSuccess ) {
        std::cerr << "Failed to invoke the kernel." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }

    const auto end { std::chrono::high_resolution_clock::now() };

    //
    // Getting result from GPU to Host
    //
    err = hipMemcpy(__mask.data(), deviceMask, g_Width * g_Height * sizeof(unsigned char), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }

    std::ofstream output( "../photos/image_output.raw", std::ios::binary );
    if (!output.is_open()) {
        std::cout << "Cant open a file!\n";
        (void)hipFree(deviceBuffer);
        (void)hipFree(deviceMask);
        (void)hipFree(device_sum_00);
        (void)hipFree(device_sum_01);
        (void)hipFree(device_sum_10);
        return 1;
    }
    output.write(reinterpret_cast<const char*>(__mask.data()), __mask.size());
    output.close();

    //
    // Doing clean ups
    //
    (void)hipFree(deviceBuffer);
    (void)hipFree(deviceMask);
    (void)hipFree(device_sum_00);
    (void)hipFree(device_sum_01);
    (void)hipFree(device_sum_10);

    //
    // Just for seeing result in jpg format
    //
    stbi_write_jpg("output.jpg", g_Width, g_Height, 1, __mask.data(), 90);

    // Checking if we need the CPU test
    if (std::stoi(argv[2]) == 1) {
        // Checking the result with CPU implementation
        HandVision(__buffer, __mask_for_CPU_test);
        double result = checkResults(__mask, __mask_for_CPU_test);
        std::cout << result << " %\n";        
    }

    //
    // Printing Benchmark timer result
    //
    std::cout << "Time used: " << std::chrono::duration<double>(end - start).count() << std::endl;
    return 0;

    //
    // We good
    //
}



// CPU implementation
void HandVision(std::vector<unsigned char>& vec, std::vector<unsigned char>& mask) {
    int W         = g_Width;
    int H         = g_Height;
    int Y_size    = W * H;

    // The result may differ from one image to another, color skin
    // TODO: average skin colour
    unsigned char colR { 143 };
    unsigned char colG { 103 };
    unsigned char colB { 80 };
    unsigned char tolerance { 29 };

    size_t sum_00 { 0 };
    size_t sum_10 { 0 };
    size_t sum_01 { 0 };

    for (size_t y { 0 }; y < H; ++y) {
        for (size_t x { 0 }; x < W; ++x) {

            // brightness index
            int Y = vec[y * W + x];
            // Calculating index
            int uv_row = y / 2;
            int uv_col = x / 2;
            int uv_index = Y_size + (uv_row * W) + (uv_col * 2);
            // UV for the pixel (i,j)
            int U   = vec[uv_index];
            int V   = vec[uv_index + 1];

            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;

            int R = (298 * C           + 409 * E + 128) >> 8;
            int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
            int B = (298 * C + 516 * D           + 128) >> 8;
            
            if (R < 0) R = 0; if (R > 255) R = 255;
            if (G < 0) G = 0; if (G > 255) G = 255;
            if (B < 0) B = 0; if (B > 255) B = 255;

            if (std::abs(R - colR) <= tolerance && std::abs(G - colG) <= tolerance && std::abs(B - colB) <= tolerance) {
                mask[y * W + x] = 255;
                ++sum_00;
                sum_10 += x;
                sum_01 += y;
            } else {
                mask[y * W + x] =  0;
            }
        }
    }

    size_t x_coord { sum_10 / sum_00 };
    size_t y_coord { sum_01 / sum_00 };

    int r { 1100 };
    int d { 1350 };

    for (int dx = -r; dx <= r; ++dx) {
        int nx = x_coord + dx;

        int ny1 = y_coord - d;
        int ny2 = y_coord + d;

        if (nx >= 0 && nx < W) {
            if (ny1 >= 0 && ny1 < H) {
                mask[ny1 * W + nx] = 255;
            }
            if (ny2 >= 0 && ny2 < H) {
                mask[ny2 * W + nx] = 255;
            }
        }
    }

    for (int dy = -d; dy <= d; ++dy) {
        int ny = y_coord + dy;

        int nx1 = x_coord - r;
        int nx2 = x_coord + r;

        if (ny >= 0 && ny < H) {
            if (nx1 >= 0 && nx1 < W) {
                mask[ny * W + nx1] = 255;
            }
            if (nx2 >= 0 && nx2 < W) {
                mask[ny * W + nx2] = 255;
            }
        }
    }
}


double checkResults(const std::vector<unsigned char>& __mask_fromGPU, const std::vector<unsigned char>& __mask_for_CPU_test) {
    // Image size
    const size_t image_size{ g_Height * g_Width };
    size_t correct_values { 0 };

    // Checking that every pixel is correct
    for (size_t i = 0; i < __mask_fromGPU.size(); ++i) {
        if (__mask_fromGPU[i] == __mask_for_CPU_test[i]) {
            ++correct_values;
        } else {
            continue;
        }
    }

    // Returning the Result
    return correct_values / image_size;
}