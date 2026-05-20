#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <iterator>

constexpr int g_Width { 6720 };
constexpr int g_Height { 4480 };

// How to convert jpg image to nv12 format:
// usage: ffmpeg -i 2.jpg -pix_fmt nv12 -f rawvideo output.nv12

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

//
// This is CPU implementation
//
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "usage: ./app.exe <path>" << std::endl;
        return 1;
    }

    const auto start { std::chrono::high_resolution_clock::now() };

    std::ifstream input( argv[1], std::ios::binary | std::ios::ate );
    if (!input.is_open()) {
        std::cout << "Cant open a file!\n";
        return 1;
    }

    std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (size <= 0) {
        std::cout << "File is empty!" << std::endl;
        return 1;
    }

    std::vector<unsigned char> buffer(size);
    if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cout << "unable to copy data to vector!" << std::endl;
        return 1;
    }

    std::vector<unsigned char> mask(g_Width * g_Height);
    input.close();

    HandVision(buffer, mask);

    std::ofstream output( "../photos/image_output.raw", std::ios::binary );
    if (!output.is_open()) {
        std::cout << "Cant open a file!\n";
        return 1;
    }
    output.write(reinterpret_cast<const char*>(mask.data()), mask.size());
    output.close();

    const auto end { std::chrono::high_resolution_clock::now() };
    
    // Just for seeing result in jpg format
    stbi_write_jpg("output.jpg", g_Width, g_Height, 1, mask.data(), 100);
    
    std::cout << "Time used: " << std::chrono::duration<double>(end - start) << std::endl;
}
