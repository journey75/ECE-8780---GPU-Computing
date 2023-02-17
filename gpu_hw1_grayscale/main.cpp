#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <cstdio> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <iomanip>


#include "im2Gray.h"



/*
 Driver program to test im2gray
*/


/*
 Process input image and allocate memory on host and 
 GPUs.
*/


void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems){

    for(int i = 0; i < numElems; i++){
        if(ref[i] - gpu[i] > 1e-5){
            std::cerr << "Error at position " << i << "\n"; 

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] <<"\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}



void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img; 

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);


    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows*ref_img.cols*ref_img.channels());
    std::cout << "PASSED!";


}



int main(int argc, char const *argv[])
{
   uchar4 *h_imrgba, *d_imrgba;
   unsigned char *h_grey, *d_grey;

   std::string infile; 
   std::string outfile; 
   //std::string reference = "lena_gray.png";


   switch(argc){
       case 2:
            infile = std::string(argv[1]);
            outfile = "cinque_gpu_gray.png";
            break; 
        case 3:
            infile = std::string(argv[1]);
            outfile = std::string(argv[2]);
            break;
        default: 
              std::cerr << "Usage ./gray <in_image> <out_image>\n";
              exit(1);

   }

   // preprocess 

   cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR);
   cv::Mat imrgba, grayimage; 

   cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);

   grayimage.create(img.rows, img.cols, CV_8UC1);

   const size_t  numPixels = img.rows*img.cols; 

   h_imrgba = imrgba.ptr<uchar4>(0);
   h_grey = grayimage.ptr<unsigned char>(0);

   checkCudaErrors(cudaMalloc((void**)&d_imrgba, sizeof(uchar4)*numPixels));
   checkCudaErrors(cudaMalloc((void**)&d_grey, sizeof(unsigned char)*numPixels));

   checkCudaErrors(cudaMemcpy(d_imrgba, h_imrgba, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice)); 


   // call the kernel 
   launch_im2gray(d_imrgba, d_grey, img.rows, img.cols);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());

   std::cout << "Finished kernel launch \n";

   checkCudaErrors(cudaMemcpy(h_grey, d_grey, numPixels*sizeof(unsigned char), cudaMemcpyDeviceToHost));

   // create the image with the output data 

   cv::Mat output(img.rows, img.cols, CV_8UC1, (void*)h_grey);

   bool suc = cv::imwrite(outfile.c_str(), output);
   if(!suc){
       std::cerr << "Couldn't write image!\n";
       exit(1);
   }


   // check if the caclulation was correct to a degree of tolerance

   // checkResult(reference, outfile, 1e-5);

   cudaFree(d_imrgba);
   cudaFree(d_grey);

    return 0;
}



