# A Canny Computer Vision Edge Detector

- Using modern C++
- GPU side NVIDIA CUDA for quick optimization of the algorithm

# How to Start CPU side
- Put your image in the "photos" folder
- Then in the ```main.cpp``` change the path to the file
- After, to run the code write in console
```
mkdir build
cmake ..
ninja
./app.exe
```

# Benchmark

## __CPU__
- 2k Image input- 0.034s ~ 33 fps
- 8k Image input - 0.093s ~ 10 fps

## __GPU__
- 2k Image input - 0.0018s ~ 550 fps
- 8k Image input - 0.0042s ~ 240 fps