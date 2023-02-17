## Convert Image to GreyScale 


In this assignment you'll be required to implement the "Map" Parallel Paradigm using CUDA Kernels. The goals of this Problem are two fold:

1. Enable you to understand the CUDA Primitives and basic launch configuration patterns. 
2. Implement a **working** kernel and profile it's running time with different grid/block patterns. 

Specifically, you will be implementing a CUDA kernel to accept an input image and convert it to grayscale. All starter code has been provided, you need to fill in the kernel in `im2Gray.cu`. You will additionally require to configure the launch parameters. The code has automatic inbuilt checking built in.



### Deliverables:

This project requires a **correct** working kernel. In addition a LaTeX generated report detailing the design and the profiling results are necessary. Additionally, challenges faced during the coding of the kernel must be provided. 

### Setup on Palmetto: 

There's no setup required. Copy this folder to your home directory and request an interactive job with a GPU like so:
`qsub -I -l select=1:ncpus=20:ngpus=1:mem=16gb:gpu_model=p100,walltime=48:00:00`. 

This shall give you a node with a GPU and `nvcc`. Use the Makefile provided to compile your code.



### Local Setup:

You will need CUDA 9.0 or above, OpenCV4 and access to linux shell. If you're using Visual Studio, then you shall need to install CUDA Toolkit and OpenCV4 and integrate them in your project. It is advised you take advantage of Palmetto to avoid significant delays.
