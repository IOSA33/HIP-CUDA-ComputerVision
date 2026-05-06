cmake_minimum_required(VERSION 3.10)
project(proteinPrediction VERSION 0.1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(app main.cpp)
set(CMAKE_CXX_FLAGS "-O3 -Wall -march=native")