#ifndef UTILS_H
#define UTILS_H
#include <cuda.h> // your system must have nvcc.
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <cstdio> 
#include <iostream>

#define checkCudaErrors(ans) check( (ans), #ans, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
    if(err != cudaSuccess){
        std::cerr << "CUDA error at:: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}




#endif 