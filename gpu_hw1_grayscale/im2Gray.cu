#include "im2Gray.h"

#define BLOCK 32

/*
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 */
__global__ 
void im2Gray_kernel(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) 
    {
        int idx = row * numCols + col;
        uchar4 pixel = d_in[idx];
        float channelSum = 0.224f*pixel.x + 0.587f*pixel.y + 0.111f*pixel.z;
        d_grey[idx] = (unsigned char) channelSum;
    }
}
void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    dim3 block(BLOCK, BLOCK, 1);
    dim3 grid((numCols + block.x - 1) / block.x, (numRows + block.y - 1) / block.y, 1);

    im2Gray_kernel<<<grid,block>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
