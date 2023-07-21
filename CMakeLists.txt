cmake_minimum_required(VERSION 3.13)  # CMake version check
project(rasterizer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(SDL2 REQUIRED)
find_package(CUDA REQUIRED)

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES src/main.cpp src/window.cpp src/window.cu)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(rasterizer ${SOURCE_FILES})
target_include_directories(rasterizer PUBLIC include ${SDL2_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(rasterizer ${SDL2_LIBRARIES} ${CUDA_LIBRARIES})
