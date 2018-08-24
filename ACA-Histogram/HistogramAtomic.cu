#include <stdio.h>
#include <stdint.h>
#include <cutil_inline.h>
#include "cutil.h"
#include "device_functions.h"
#include <time.h>
#include <sys/time.h>

#define H 	64

// Default values
int N = 10000; 		//Size
int T = 32; 		//Blocksize
int B = 4; 		//Blocks

// Host Variables
int* HostData;
int* HostHist;
int* HostTimer=NULL;

// Device Variables
int* DeviceData;
int* DeviceHist;
int* DeviceTimer=NULL;

// Timer Variables
struct timeval CPU_Time_start, CPU_Time_end;
struct timeval GPU_Time_start, GPU_Time_end;
struct timeval DeviceToHost_start, DeviceToHost_end;
struct timeval HostToDevice_start, HostToDevice_end;
struct timeval CPU_Partial_Time_start, CPU_Partial_Time_end;
struct timeval CPU_Cleanup_Time_start, CPU_Cleanup_Time_end;
struct timeval Total_Time_start, Total_Time_end;

// Function Declaration
void Cleanup(void);
void HistogramSequential(int* result, int* data, int size);


// Histogram Atomic Kernel
__global__ void histogram_kernel(int* PartialHist, int* DeviceData, int DataCount,int* timer)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;    
    int stride = blockDim.x * gridDim.x; 
    clock_t start_atomic=0;
    clock_t stop_atomic=0;

    extern __shared__ int hist[];

    if(tid==0)
    {
        start_atomic = clock();
    }
	
    for(int i = 0; i< H; i++)
        hist[tid * H + i] = 0; 

    for(int j = gid; j < DataCount; j += stride)
        hist[tid * H + DeviceData[j]]++;

    __syncthreads();    

    for(int t_hist = 0; t_hist < blockDim.x; t_hist++)
    {
        atomicAdd(&PartialHist[tid],hist[t_hist * H + tid]);
        atomicAdd(&PartialHist[tid + blockDim.x],hist[t_hist * H + tid + blockDim.x]);
    }
    stop_atomic=clock();

    if(tid==0)
    {
        timer[blockIdx.x] = stop_atomic - start_atomic;
    }   
}


int main(int argc, char** argv)
{

    int CPU_Hist[64] = {0};
    int GPU_Hist[64] = {0};	
    int SharedMemorySize = T * H * sizeof(int);
    int i,j;
    int RESULT=1; 

   // Memory allocation variables.
    size_t DataSize = N * sizeof(int);
    size_t AtomicHistSize = sizeof(int) * H;
    size_t HistSize = B * H * sizeof(int);
    size_t TimerSize = 2 * B * sizeof(int);

    printf("\nHistogram Kernel with atomic add function\n");    
    printf("\nACA-Histogram\nSize: %d\nThreads per Block: %d\nBlocks: %d\n", N ,T ,B);

    //Host memory alloation
    HostData = (int*)malloc(DataSize);
    HostTimer = (int*)malloc(TimerSize);
    HostHist = (int*)malloc(AtomicHistSize);

    //device memory allocation
    cutilSafeCall( cudaMalloc((void**)&DeviceData, DataSize) );
    cutilSafeCall( cudaMalloc((void**)&DeviceTimer, TimerSize));
    cutilSafeCall( cudaMalloc((void**)&DeviceHist, AtomicHistSize) );
    cutilSafeCall( cudaMemset(DeviceHist, 0, AtomicHistSize));

    //Initialize the datavalues
    srand(1); //set rand() seed to 1 for repeatability
    for (i = 0; i < N; ++i) 
        HostData[i] = rand() % H;

    //CPU Histogram Execution
    //cutStartTimer(CPU_Time);
    gettimeofday(&CPU_Time_start,NULL);
    HistogramSequential(CPU_Hist, HostData, N);
    gettimeofday(&CPU_Time_end,NULL);
    //cutStopTimer(CPU_Time);

    //Allocate device memory for data
    //cutStartTimer(Total_Time);
    //cutStartTimer(HostToDevice);
    gettimeofday(&Total_Time_start,NULL);
    gettimeofday(&HostToDevice_start,NULL);
    cutilSafeCall(cudaMemcpy(DeviceData, HostData, DataSize, cudaMemcpyHostToDevice));
    gettimeofday(&HostToDevice_end,NULL);
    //cutStopTimer(HostToDevice);

    //Call GPU Kernel
    //cutStartTimer(GPU_Time);
    gettimeofday(&GPU_Time_start,NULL);
    histogram_kernel<<<B,T,SharedMemorySize>>>(DeviceHist,DeviceData,N,DeviceTimer);
    gettimeofday(&GPU_Time_end,NULL);
    //cutStopTimer(GPU_Time);

    cutilSafeCall(cudaMemcpy(HostTimer, DeviceTimer, TimerSize, cudaMemcpyDeviceToHost));

    //cutStartTimer(DeviceToHost);
    gettimeofday(&DeviceToHost_start,NULL);
    cutilSafeCall(cudaMemcpy(HostHist, DeviceHist, AtomicHistSize, cudaMemcpyDeviceToHost));
    gettimeofday(&DeviceToHost_end,NULL);
    gettimeofday(&Total_Time_end,NULL);
    //cutStopTimer(DeviceToHost);
    //cutStopTimer(Total_Time);

    //cutStartTimer(CPU_Partial_Time);
    gettimeofday(&CPU_Partial_Time_start,NULL);
    for(j = 0; j < H; j++)
    	GPU_Hist[j] = HostHist[j];
    gettimeofday(&CPU_Partial_Time_end,NULL);
    //cutStopTimer(CPU_Partial_Time);

    //Print timer values for blocks
    for(j=0; j < B; j++)
    {
    	printf("\nTime for Atomic Block %d: %d\n",j,HostTimer[j]);
    }

    //Compare the GPU and CPU results    
    for(i = 0; i< H; i++)
    {
        if(GPU_Hist[i] != CPU_Hist[i])
        {
            RESULT=0;
        }
    }

    if(RESULT)
        printf("\nMatching Histogram\n");
    else
        printf("\nError in Results\n");

    printf("\nCPU Execution time = %ld\n", (CPU_Time_end.tv_usec-CPU_Time_start.tv_usec));

    printf("\nGPU Kernel Time = %ld\n",(GPU_Time_end.tv_usec-GPU_Time_start.tv_usec));
    printf("Memory Transfer from Host to Device = %ld\n",(HostToDevice_end.tv_usec-HostToDevice_start.tv_usec));
    printf("Memory Transfer from Device to Host = %ld\n",(DeviceToHost_end.tv_usec-DeviceToHost_start.tv_usec));
    printf("CPU Partial Time = %ld\n", (CPU_Partial_Time_end.tv_usec-CPU_Partial_Time_start.tv_usec));
    printf("GPU Total Execution Time = %ld\n",(Total_Time_end.tv_usec-Total_Time_start.tv_usec));


    Cleanup();
}


void HistogramSequential(int* result, int* data, int size)
{
    for(int i=0; i<H; i++)
        result[i]=0;

    for(int j=0; j<size; j++)
        result[data[j]]++;
}


void Cleanup(void)
{
    gettimeofday(&CPU_Cleanup_Time_start,NULL);
    if(DeviceData)
       cudaFree(DeviceData);

    if(DeviceHist)
       cudaFree(DeviceHist);

    gettimeofday(&CPU_Cleanup_Time_end,NULL);   

    if(DeviceTimer)
       cudaFree(DeviceTimer);

    if(HostData)
       free(HostData);

    if(HostHist)       
       free(HostHist); 
       
    if(HostTimer)
       free(HostTimer);
    
    printf("\nCPU Cleanup Time = %ld\n",(CPU_Cleanup_Time_end.tv_usec-CPU_Cleanup_Time_start.tv_usec));

    cutilDeviceReset();
}

