cmake_minimum_required(VERSION 3.23)
project(image_gray LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 60;70;80;86)

find_package( OpenCV REQUIRED )

add_executable(main main.cpp im2Gray.cu)
target_link_libraries(main cufft opencv_core opencv_imgcodecs opencv_highgui)
