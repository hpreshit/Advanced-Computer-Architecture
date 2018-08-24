// 
// Vector Reduction
//

// Includes
#include <stdio.h>
#include <stdint.h>
#include <cutil_inline.h>
#include "cutil.h"
#include "device_functions.h"

// Input Array Variables
float* h_In = NULL;
float* d_In = NULL;

// Output Array
float* h_Out = NULL;
float* d_Out = NULL;

// Variables to change
int GlobalSize = 1024;
int BlockSize = 32;

// Variables for Timers
uint32_t GPU_Time = 0;
uint32_t CPU_Time = 0;
uint32_t HostToDevice = 0;
uint32_t DeviceToHost = 0;
uint32_t Total_Time = 0;

// Device Variables
__device__ float d_Total = 0.0;

// Functions
void Cleanup(void);
void RandomInit(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);

// Device code
__global__ void VecReduce(float* g_idata, float* g_odata, int N)
{
  // shared memory size declared at kernel launch
  extern __shared__ float sdata[]; 

  unsigned int tid = threadIdx.x; 
  unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x; 

  // For thread ids greater than data space
  if (globalid < N) {
     sdata[tid] = g_idata[globalid]; 
  }
  else {
     sdata[tid] = 0;  // Case of extra threads above N
  }

  // each thread loads one element from global to shared mem
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
     if (tid < s) { 
         sdata[tid] = sdata[tid] + sdata[tid+ s];
     }
     __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)  {
     //g_odata[blockIdx.x] = sdata[0];
     atomicAdd(&d_Total,sdata[tid]);
  }
}


// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    int N = GlobalSize;
    printf("Vector reduction: size %d\n", N);
    size_t in_size = N * sizeof(float);
    float CPU_result = 0.0, GPU_result = 0.0;

    //Create Timers
    cutCreateTimer(&CPU_Time);
    cutCreateTimer(&GPU_Time);
    cutCreateTimer(&HostToDevice);
    cutCreateTimer(&DeviceToHost);
    cutCreateTimer(&Total_Time);

    // Allocate input vectors h_In and h_B in host memory
    h_In = (float*)malloc(in_size);
    if (h_In == 0) 
      Cleanup();

    // Initialize input vectors
    RandomInit(h_In, N);

    // Set the kernel arguments
    int threadsPerBlock = BlockSize;
    int sharedMemSize = threadsPerBlock * sizeof(float);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t out_size = blocksPerGrid * sizeof(float);

    // Allocate host output
    h_Out = (float*)malloc(out_size);
    if (h_Out == 0) 
      Cleanup();

    // STUDENT: CPU computation - time this routine for base comparison
    CPU_result = CPUReduce(h_In, N);

    // Allocate vectors in device memory
    cutilSafeCall( cudaMalloc((void**)&d_In, in_size) );
    cutilSafeCall( cudaMalloc((void**)&d_Out, out_size) );
    
    // STUDENT: Copy h_In from host memory to device memory
    cutStartTimer (Total_Time);
    cutStartTimer (HostToDevice);
    cutilSafeCall (cudaMemcpy(d_In,h_In,in_size,cudaMemcpyHostToDevice));
    cutStopTimer (HostToDevice);

    // Invoke kernel
    cutStartTimer (GPU_Time);
    VecReduce<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_In, d_Out, N);
    cutilCheckMsg("kernel launch failure");
    cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
    cutStopTimer (GPU_Time);

    // STUDENT: copy results back from GPU to the h_Out
    cutStartTimer (DeviceToHost);
    //cutilSafeCall( cudaMemcpy(h_Out,d_Out,out_size,cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaMemcpyFromSymbol(&GPU_result,"d_Total",sizeof(float),0,cudaMemcpyDeviceToHost));
    cutStopTimer (DeviceToHost);

    // STUDENT: Perform the CPU addition of partial results
    // update variable GPU_result
    /*cutStartTimer (CPU_Time);
    for(int i=0;i<blocksPerGrid;i++)
    {
	GPU_result=GPU_result+h_Out[i];
    }
    cutStopTimer (CPU_Time);*/
    cutStopTimer (Total_Time);
   
    // STUDENT Check results to make sure they are the same
    printf("CPU results : %f\n", CPU_result);
    printf("GPU results : %f\n", GPU_result);

    printf("GPU Execution Time = %f\n",cutGetTimerValue(GPU_Time));
    printf("Memory Transfer from Host to Device = %f\n",cutGetTimerValue(HostToDevice));
    printf("Memory Transfer from Device to Host = %f\n",cutGetTimerValue(DeviceToHost));
    printf("Memory Transfer Time = %f\n",cutGetTimerValue(HostToDevice)+cutGetTimerValue(DeviceToHost));
    //printf("CPU Time to add partial sums = %f\n", cutGetTimerValue(CPU_Time));
    printf("Total Time = %f\n",cutGetTimerValue(Total_Time));
 
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_In)
        cudaFree(d_In);
    if (d_Out)
        cudaFree(d_Out);

    // Free host memory
    if (h_In)
        free(h_In);
    if (h_Out)
        free(h_Out);
        
    cutilSafeCall( cudaThreadExit() );
    
    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}

void PrintArray(float* data, int n)
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

float CPUReduce(float* data, int n)
{
  float sum = 0;
    for (int i = 0; i < n; i++)
        sum = sum + data[i];

  return sum;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  GlobalSize = atoi(argv[i+1]);
		  i = i + 1;
        }
        if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
                  BlockSize = atoi(argv[i+1]);
		  i = i + 1;
	}
    }
}
