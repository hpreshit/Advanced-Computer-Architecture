#include <stdio.h>
#include <stdint.h>
#include <cutil_inline.h>
#include "cutil.h"
#include <time.h>
#include <sys/time.h>

#define H 	64

// Default values
int N = 10000; 		//Size
int T = 32; 		//BlockSize
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

// Histogram kernel
__global__ void histogram_kernel(int* PartialHist, int* DeviceData, int dataCount,int* timer)
{   
    unsigned int tid = threadIdx.x; 
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;    
    unsigned int stride = blockDim.x * gridDim.x; 
    clock_t start_clock=0;
    clock_t stop_clock=0;

    if(tid==0)
    {
        start_clock = clock();
    }

     __shared__ int BlockHist[H];

    extern __shared__ int hist[];

    for(int h = 0; h < H; h++)
    {    
        hist[tid * H + h]=0;
    }

    BlockHist[tid] = 0;
    BlockHist[tid + blockDim.x] = 0;

    for(int pos = gid; pos < dataCount; pos += stride)
        hist[tid * H + DeviceData[pos]]++; 

    for(int t_hist = 0; t_hist < blockDim.x; t_hist++)
    {
        BlockHist[tid] += hist[t_hist * H + tid];
        BlockHist[tid+blockDim.x] += hist[(t_hist * H)+(tid + blockDim.x)];
    }

    PartialHist[tid+(blockIdx.x * H)] = BlockHist[tid];
    PartialHist[tid+(blockIdx.x * H) + blockDim.x] = BlockHist[tid + blockDim.x];

    if(tid==0)
    {
        stop_clock = clock();
        timer[blockIdx.x * 2] = start_clock;
        timer[blockIdx.x * 2 + 1] = stop_clock;
    }
}


int main(int argc, char** argv)
{
    int CPU_Hist[64] = {0};
    int GPU_Hist[64] = {0};	
    int SharedMemorySize = H * T  * sizeof(int);
    int i,j,BlockNum;
    int RESULT=1;

   // Memory allocation variables.
    size_t DataSize = N * sizeof(int);
    size_t HistSize = B * H * sizeof(int);
    size_t TimerSize = 2 * B * sizeof(int);


    printf("\nACA-Histogram\nSize: %d\nThreads Per Block: %d\nBlocks: %d\n", N, T, B);   

    //Host memory alloation
    HostData = (int*)malloc(DataSize);
    HostTimer = (int*)malloc(TimerSize);
    HostHist = (int*)malloc(HistSize);

    //device memory allocation
    cutilSafeCall( cudaMalloc((void**)&DeviceData, DataSize) );
    cutilSafeCall( cudaMalloc((void**)&DeviceTimer, TimerSize));
    cutilSafeCall( cudaMalloc((void**)&DeviceHist, HistSize) ); 
    cutilSafeCall( cudaMemset(DeviceHist, 0, HistSize));

    //Initialize the datavalues
    srand(1); //set rand() seed to 1 for repeatability
    for (i = 0; i < N; ++i) 
        HostData[i] = rand() % H;

    //CPU Histogram Execution
    //cutStartTimer(CPU_Time);
    gettimeofday(&CPU_Time_start,NULL);
    HistogramSequential(CPU_Hist, HostData, N);
    //cutStopTimer(CPU_Time);
    gettimeofday(&CPU_Time_end,NULL);

    //Allocate device memory for data
    //cutStartTimer(Total_Time);
    gettimeofday(&Total_Time_start,NULL);
    //cutStartTimer(HostToDevice);
    gettimeofday(&HostToDevice_start,NULL);
    cutilSafeCall(cudaMemcpy(DeviceData, HostData, DataSize, cudaMemcpyHostToDevice));
    //cutStopTimer(HostToDevice);
    gettimeofday(&HostToDevice_end,NULL);

    //Call GPU Kernel	
    //cutStartTimer(GPU_Time);
    gettimeofday(&GPU_Time_start,NULL);
    histogram_kernel<<<B,T,SharedMemorySize>>>(DeviceHist,DeviceData,N,DeviceTimer);
    //cutStopTimer(GPU_Time);
    gettimeofday(&GPU_Time_end,NULL);
    cutilSafeCall(cudaMemcpy(HostTimer, DeviceTimer, TimerSize, cudaMemcpyDeviceToHost));

    //Memory Transfer from Device to Host
    //cutStartTimer(DeviceToHost);
    gettimeofday(&DeviceToHost_start,NULL);
    cutilSafeCall(cudaMemcpy(HostHist, DeviceHist, HistSize, cudaMemcpyDeviceToHost));
    //cutStopTimer(DeviceToHost);
    gettimeofday(&DeviceToHost_end,NULL);
    //cutStopTimer(Total_Time);
    gettimeofday(&Total_Time_end,NULL);

    //cutStartTimer(CPU_Partial_Time);
    gettimeofday(&CPU_Partial_Time_start,NULL);
    for(BlockNum = 0; BlockNum < B; BlockNum++)
    {
   	for(i = 0; i < H; i++)
	    GPU_Hist[i] += HostHist[i +(BlockNum * H)];
    }
    //cutStopTimer(CPU_Partial_Time);
    gettimeofday(&CPU_Partial_Time_end,NULL);

    //Print timer values for blocks
    for(j=0; j < B; j++)
    {
        printf("\nBlock Number: %d \nStart time: %d\nStop time: %d\nDifference: %d\n",\
	    j,HostTimer[j*2],HostTimer[j*2 + 1],(HostTimer[j*2+1] - HostTimer[j*2]));
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
    //cutStartTimer(CPU_Cleanup_Time);
    gettimeofday(&CPU_Cleanup_Time_start,NULL);
    if(DeviceData)
       cudaFree(DeviceData);

    if(DeviceHist)
       cudaFree(DeviceHist);
    
    //cutStopTimer(CPU_Cleanup_Time);
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
    
    //cutDeleteTimer(CPU_Cleanup_Time);
    cutilDeviceReset();
}
