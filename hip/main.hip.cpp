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
// Predefined photo size
//
constexpr int g_Width { 3072 };
constexpr int g_Height { 4096 };

__global__ void DrawRectangle(unsigned long long sum_00, unsigned long long sum_10, unsigned long long sum_01) {
    unsigned long long x_coord { sum_10 / sum_00 };
    unsigned long long y_coord { sum_01 / sum_00 };

    int r { 1100 };
    int d { 1350 };

    for (int dx = -r; dx <= r; ++dx) {
        int nx = x_coord + dx;

        int ny1 = y_coord - d;
        int ny2 = y_coord + d;

        if (nx >= 0 && nx < width) {
            if (ny1 >= 0 && ny1 < height) {
                mask[ny1 * width + nx] = 255;
            }
            if (ny2 >= 0 && ny2 < height) {
                mask[ny2 * W + nx] = 255;
            }
        }
    }

    for (int dy = -d; dy <= d; ++dy) {
        int ny = y_coord + dy;

        int nx1 = x_coord - r;
        int nx2 = x_coord + r;

        if (ny >= 0 && ny < height) {
            if (nx1 >= 0 && nx1 < W) {
                mask[ny * W + nx1] = 255;
            }
            if (nx2 >= 0 && nx2 < W) {
                mask[ny * W + nx2] = 255;
            }
        }
    }
}

//
// Doing hand vision work with kernels
//
__global__ void HandVisionGPU(unsigned char* vec, unsigned char* mask, int width, int height) {
    int Y_size    = width * height;

    // The result may differ from one image to another, color skin
    // TODO: average skin colour
    unsigned char colR { 143 };
    unsigned char colG { 103 };
    unsigned char colB { 80 };
    unsigned char tolerance { 29 };

    unsigned long long sum_00 { 0 };
    unsigned long long sum_10 { 0 };
    unsigned long long sum_01 { 0 };

    for (unsigned long long y { 0 }; y < height; ++y) {
        for (unsigned long long x { 0 }; x < width; ++x) {

            // brightness index
            int Y = vec[y * width + x];
            // Calculating index
            int uv_row = y / 2;
            int uv_col = x / 2;
            int uv_index = Y_size + (uv_row * width) + (uv_col * 2);
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
                mask[y * width + x] = 255;
                
                ++sum_00;
                sum_10 += x;
                sum_01 += y;

            } else {
                mask[y * width + x] =  0;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "usage: ./app.exe <path_to_file>"
        return 1;
    }

    //
    // HIP error type for function checks
    //
    hipError_t err;

    //
    // Starting timer
    //
    const auto start { std::chrono::high_resolution_clock::now() };
    
    //
    // Opening file and read raw bytes from it
    //
    std::ifstream input( "../photos/output.nv12", std::ios::binary | std::ios::ate );
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
    if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "unable to copy data to vector!" << std::endl;
        return 1;
    }

    //
    // Reserving mask vector for output result
    //
    std::vector<unsigned char> __mask(g_Width * g_Height);
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
    // Allocating space in global memory
    //
    std::cout << "Allocating " << g_Width * g_Height * sizeof(unsigned char) / 1.0e6 << " MB of global memory." << std::endl;
    unsigned char* deviceBuffer;
    unsigned char* deviceMask;
    err = hipMalloc(&deviceBuffer, g_Width * g_Height * sizeof(unsigned char));
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

    //
    // Transfering raw bytes __buffer to device memory, we dont transfer __mask because
    // its empty vector and we only gonna store it there the data
    //
    std::cout << "Copying " << g_Width * g_Height * sizeof(unsigned char) / 1.0e6 << " MB from host to device." << std::endl;
    err = hipMemcpy(deviceBuffer, __buffer, g_Width * g_Height * sizeof(unsigned char), hipMemcpyHostToDevice);
    if ( err != hipSuccess ) {
        std::cerr << "Failed to copy memory to device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        return 1;
    }

    //
    // Configuring blocks and threads
    //
    const dim3 numberOfBlocks((g_Width * g_Height - 1) / 256 + 1);
    const dim3 threadsPerBlock(256);

    //
    // Kernel Call
    //
    std::cout << "Calling kernel!" << std::endl;
    HandVisionGPU<<<numberOfBlocks, threadsPerBlock, 0>>>(deviceBuffer, __mask, g_Width, g_Height);
    err = hipGetLastError();
    if ( err != hipSuccess ) {
        std::cerr << "Failed to invoke the kernel." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        return 1;
    }

    //
    // Getting the result from the GPU
    //
    std::cout << "Copying " << g_Width * g_Height * sizeof(unsigned char) / 1.0e6 << " MB from device to host." << std::endl;
    err = hipMemcpy(__mask, deviceBuffer, g_Width * g_Height * sizeof(unsigned char), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy memory from device." << std::endl;
        std::cerr << "hipError-code: " << err << std::endl;
        std::cerr << "hipError-string: " << hipGetErrorString(err) << std::endl;

        (void)hipFree(deviceBuffer);
        return 1;
    }

    std::ofstream output( "../photos/image_output.raw", std::ios::binary );
    if (!output.is_open()) {
        std::cout << "Cant open a file!\n";
        return 1;
    }
    output.write(reinterpret_cast<const char*>(mask.data()), mask.size());
    output.close();

    //
    // Doing clean ups
    //
    (void)hipFree(deviceBuffer);

    //
    // Checking How much time it took to complete
    //
    const auto end { std::chrono::high_resolution_clock::now() };

    //
    // Just for seeing result in jpg format
    //
    stbi_write_jpg("output.jpg", g_Width, g_Height, 1, mask.data(), 90);

    //
    // Printing Benchmark timer result
    //
    std::cout << "Time used: " << std::chrono::duration<double>(end - start).count() << std::endl;
    return 0;

    //
    // We good
    //
}

