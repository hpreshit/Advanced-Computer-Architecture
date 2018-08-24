#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
//#include <sys/io.h>
#include <time.h>
#include <sys/time.h>

#include <cutil_inline.h>

// Includes: local
#include "bmp.h"

enum {SOBEL_FILTER=1, AVERAGE_FILTER, HIGH_BOOST_FILTER};

#define CLAMP_8bit(x) max(0, min(255, (x)))

char *BMPInFile = "dublin.bmp";
char *BMPOutFile = "sobel5x5_dublin.bmp";
char *Filter = "sobel";
int FilterMode  = SOBEL_FILTER;

// Functions
void Cleanup(void);
void ParseArguments(int, char**);
void FilterWrapper(unsigned char* pImageIn, int Width, int Height);

//Kernel functions
__global__ void SobelFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height);
__global__ void AverageFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height);
__global__ void HighBoostFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height);


/* Device Memory */
unsigned char *d_In;
unsigned char *d_Out;
float *d_sobel;

// Setup for kernel size
const int TILE_WIDTH    = 6;
const int TILE_HEIGHT   = 6;

const int FILTER_RADIUS = 2;
//  const int FILTER_RADIUS = 3;

const int FILTER_DIAMETER = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA   = FILTER_DIAMETER * FILTER_DIAMETER;

const int BLOCK_WIDTH   = TILE_WIDTH + 2*FILTER_RADIUS;
const int BLOCK_HEIGHT  = TILE_HEIGHT + 2*FILTER_RADIUS;

const int EDGE_VALUE_THRESHOLD = 800;
const int HIGH_BOOST_FACTOR = 10;

// Variables for Timers

struct timeval CPU_Time_start, CPU_Time_end;
struct timeval GPU_Time_start, GPU_Time_end;
struct timeval HostToDevice_start, HostToDevice_end;
struct timeval DeviceToHost_start, DeviceToHost_end;

/*unsigned int CPU_Time = 0;		
unsigned int GPU_Time = 0;		
unsigned int HostToDevice = 0;	
unsigned int DeviceToHost = 0;*/    	


#include "filter_kernel.cu"

void BitMapRead(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char **data, unsigned char **palete)
{
   size_t palete_size;
   int fd;

   if((fd = open(file, O_RDONLY )) < 0)
           FATAL("Open Source");

   if(read(fd, bmp, BMP_SIZE) != BMP_SIZE)
           FATAL("Read BMP Header");

   if(read(fd, dib, DIB_SIZE) != DIB_SIZE)
           FATAL("Read DIB Header");

   assert(dib->bpp == 8);

   palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
   if(palete_size > 0) {
           *palete = (unsigned char *)malloc(palete_size);
           int go = read(fd, *palete, palete_size);
           if (go != palete_size) {
                   FATAL("Read Palete");
           }
   }

   *data = (unsigned char *)malloc(dib->image_size);
   if(read(fd, *data, dib->image_size) != dib->image_size)
           FATAL("Read Image");

   close(fd);
}


void BitMapWrite(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char *data, unsigned char *palete)
{
   size_t palete_size;
   int fd;

   palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;

   if((fd = open(file, O_WRONLY | O_CREAT | O_TRUNC,
                             S_IRUSR | S_IWUSR |S_IRGRP)) < 0)
           FATAL("Open Destination");

   if(write(fd, bmp, BMP_SIZE) != BMP_SIZE)
           FATAL("Write BMP Header");

   if(write(fd, dib, DIB_SIZE) != DIB_SIZE)
           FATAL("Write BMP Header");

   if(palete_size != 0) {
           if(write(fd, palete, palete_size) != palete_size)
                   FATAL("Write Palete");
   }
   if(write(fd, data, dib->image_size) != dib->image_size)
           FATAL("Write Image");
   close(fd);
}



void CPU_Sobel(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  const float SobelMatrix[25] = {1,2,0,-2,-1,4,8,0,-8,-4,6,12,0,-12,-6,4,8,0,-8,-4,1,2,0,-2,-1};

  rows = height;
  cols = width;
 
  // Initialize all output pixels to zero 
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
	imageOut[i*width + j] = 0;
    }
  }

  startCol = 1;
  endCol = cols - 1;
  startRow = 1;
  endRow = rows - 1;
  
  // Go through all inner pizel positions 
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {

       // sum up the 25 values to calculate both the direction x and direction y
       float sumX = 0, sumY=0;
       for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
          for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             float Pixel = (float)(imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel * SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
             sumY += Pixel * SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
          }
	}
       imageOut[i*width + j] = (abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0;
    }
  }
}


// Host code
int main(int argc, char** argv)
{
    ParseArguments(argc, argv);

    struct bmp_header bmp;
    struct dib_header dib;

    /*Timer Create*/
    /*cutCreateTimer(&CPU_Time);
    cutCreateTimer(&GPU_Time);
    cutCreateTimer(&HostToDevice);
    cutCreateTimer(&DeviceToHost);*/

    unsigned char *palete = NULL;
    unsigned char *data = NULL, *out = NULL;
    float sobel_5x5[FILTER_AREA]={1,2,0,-2,-1,4,8,0,-8,-4,6,12,0,-12,-6,4,8,0,-8,-4,1,2,0,-2,-1};
	
    printf("Running %s filter\n", Filter);
    BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
    out = (unsigned char *)malloc(dib.image_size);

    printf("Computing the CPU output\n");
    printf("Image details: %d by %d = %d , imagesize = %d\n", dib.width, dib.height, dib.width * dib.height,dib.image_size);
    //cutStartTimer(CPU_Time); 		//Start timer for CPU_Time
    gettimeofday(&CPU_Time_start,NULL);                           
    CPU_Sobel(data, out, dib.width, dib.height);
    //cutStopTimer(CPU_Time); 		//Stop timer for CPU_Time
    gettimeofday(&CPU_Time_end,NULL);                           
    BitMapWrite("CPU_sobel5x5_dublin.bmp", &bmp, &dib, out, palete);
    printf("Done with CPU output\n");

    printf("Allocating %d bytes for image \n", dib.image_size);
    cutilSafeCall( cudaMalloc( (void **)&d_In, dib.image_size*sizeof(unsigned char)) );
    cutilSafeCall( cudaMalloc( (void **)&d_Out, dib.image_size*sizeof(unsigned char)) );
    cutilSafeCall( cudaMalloc( (void **)&d_sobel, FILTER_AREA*sizeof(float)));
	
    //cutStartTimer(HostToDevice); 	//Start timer for Memory Transfer from Host to Device
    gettimeofday(&HostToDevice_start,NULL);                           
    cudaMemcpy(d_In, data, dib.image_size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel, sobel_5x5, FILTER_AREA*sizeof(float), cudaMemcpyHostToDevice);
    gettimeofday(&HostToDevice_end,NULL); 
    //cutStopTimer(HostToDevice); 	//Start timer for Memory Transfer from Host to Device

    //cutStartTimer(GPU_Time); 		//Start Timer for GPU_Kernel
    gettimeofday(&GPU_Time_start,NULL);                           
    FilterWrapper(data, dib.width, dib.height);
    gettimeofday(&GPU_Time_end,NULL);                           
    //cutStopTimer(GPU_Time);		//Stop Timer for GPU_Kernel

    // Copy image back to host
    //cutStartTimer(DeviceToHost);	//Start timer for Memory Transfer from Device to Host
    gettimeofday(&DeviceToHost_start,NULL);                           
    cudaMemcpy(out, d_Out, dib.image_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    gettimeofday(&DeviceToHost_end,NULL);                           
    //cutStopTimer(DeviceToHost);		//Stop timer for Memory Transfer from Device to Host

    // Timer Values
    printf("CPU Execution Time: %ld\n",(CPU_Time_end.tv_usec-CPU_Time_start.tv_usec));
    printf("GPU Execution Time: %ld\n",(GPU_Time_end.tv_usec-GPU_Time_start.tv_usec));
    printf("Host to Device Transfer Time: %ld\n",(HostToDevice_end.tv_usec-HostToDevice_start.tv_usec));
    printf("Device to Host Transfer Time: %ld\n",(DeviceToHost_end.tv_usec-DeviceToHost_start.tv_usec));

    // Write output image   
    BitMapWrite(BMPOutFile, &bmp, &dib, out, palete);

    Cleanup();
}

void Cleanup(void)
{
    // Delete Timers
    /*cutDeleteTimer(CPU_Time);
    cutDeleteTimer(GPU_Time);
    cutDeleteTimer(HostToDevice);
    cutDeleteTimer(DeviceToHost);*/

    cutilSafeCall( cudaThreadExit() );
    exit(0);
}


void FilterWrapper(unsigned char* pImageIn, int Width, int Height)
{
   // Design grid disection around tile size
   int gridWidth  = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
   int gridHeight = (Height + TILE_HEIGHT - 1) / TILE_HEIGHT;
   dim3 dimGrid(gridWidth, gridHeight);

   // But actually invoke larger blocks to take care of surrounding shared memory
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

   switch(FilterMode) {
     case SOBEL_FILTER:
     printf("Sobel Filter \n");
     SobelFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
     cutilCheckMsg("kernel launch failure");
     break;
     case AVERAGE_FILTER:
     printf("Average Filter \n");
     AverageFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
     cutilCheckMsg("kernel launch failure");
     break;
     case HIGH_BOOST_FILTER:
     printf("Boost Filter \n");
     HighBoostFilter<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
     cutilCheckMsg("kernel launch failure");
     break;
    }
   cutilSafeCall( cudaThreadSynchronize() );
}



// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-file") == 0) {
            BMPInFile = argv[i+1];
	    i = i + 1;
        }
        if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-out") == 0) {
            BMPOutFile = argv[i+1];
	    i = i + 1;
        }
        if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-filter") == 0) {
            Filter = argv[i+1];
	    i = i + 1;
            if (strcmp(Filter, "sobel") == 0)
		FilterMode = SOBEL_FILTER;
            else if (strcmp(Filter, "average") == 0)
		FilterMode = AVERAGE_FILTER;
            else if (strcmp(Filter, "boost") == 0)
		FilterMode = HIGH_BOOST_FILTER;
	 
        }
    }
}



