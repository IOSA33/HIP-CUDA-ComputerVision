# Computer Vision Hand Detector
- Using modern C++
- GPU side AMD HIP/CUDA for quick optimization of the algorithm

## __CPU__ Test
- 2k Image input- 0.034s ~ 33 fps
- 8k Image input - 0.093s ~ 10 fps

## __GPU__ Test
- 2k Image input - 0.0018s ~ 550 fps
- 8k Image input - 0.0042s ~ 240 fps

## Image result
![Test](./hip/output.jpg)

# Reference
- https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-40-computer-vision-gpu

# How to Start CPU side
- Put your image (nv12 format) in the "photos" folder
- You can use "output.nv12" in photos folder, for testing the application
- After, to run the code write in console
```
mkdir build
cmake ..
ninja
./app.exe <path_to_nv12_file_from_build_folder>
```
# How to Start GPU side
- Put your image (nv12 format) in the "photos" folder
- You can use "output.nv12" in photos folder, for testing the application
- To run the code write in console, in hip folder
```
hipcc main.hip.cc -o app
./app.exe <path_to_nv12_file> <test?_value_1_or_0>


```