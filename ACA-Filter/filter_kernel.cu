#ifndef _FILTER_KERNEL_H_
#define _FILTER_KERNEL_H_

__global__ void SobelFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
   float s_SobelMatrix[9];

    s_SobelMatrix[0] = -1;
    s_SobelMatrix[1] = 0;
    s_SobelMatrix[2] = 1;

    s_SobelMatrix[3] = -2;
    s_SobelMatrix[4] = 0;
    s_SobelMatrix[5] = 2;

    s_SobelMatrix[6] = -1;
    s_SobelMatrix[7] = 0;
    s_SobelMatrix[8] = 1;

   // Computer the X and Y global coordinates
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 
   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();


   // STUDENT: Make sure only the thread ids should write the sum of the neighbors.
   float sumX = 0, sumY=0;
   for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++)
   {
       for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++)
       {
           float PIXEL = (float) (g_DataIn[y*width +x + (dy*width + dx)]);
           sumX += PIXEL * s_SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx + FILTER_RADIUS)];
           sumY += PIXEL * s_SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy + FILTER_RADIUS)];
       }
   }

   g_DataOut[index] = (abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0;
}


__global__ void AverageFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
   
   float AverageMatrix[9];

   AverageMatrix[0] = 1;
   AverageMatrix[1] = 1;
   AverageMatrix[2] = 1;

   AverageMatrix[3] = 1;
   AverageMatrix[4] = 1;
   AverageMatrix[5] = 1;

   AverageMatrix[6] = 1;
   AverageMatrix[7] = 1;
   AverageMatrix[8] = 1;

   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 
   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();


   // STUDENT: Make sure only the thread ids should write the sum of the neighbors.
   float sumX = 0;
   for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++)
   {
       for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++)
       {
           float PIXEL = (float) (g_DataIn[y*width + x + (dy*width + dx)]);
           sumX += PIXEL * AverageMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx + FILTER_RADIUS)];
       }
   }

   g_DataOut[index] = sumX/9;
}



__global__ void HighBoostFilter(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];

   float HighBoostMatrix[9];
   float PIXEL;

   HighBoostMatrix[0] = 1;
   HighBoostMatrix[1] = 1;
   HighBoostMatrix[2] = 1;

   HighBoostMatrix[3] = 1;
   HighBoostMatrix[4] = 1;
   HighBoostMatrix[5] = 1;

   HighBoostMatrix[6] = 1;
   HighBoostMatrix[7] = 1;
   HighBoostMatrix[8] = 1;


   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

   // Get the Global index into the original image
   int index = y * (width) + x;

   // STUDENT:  Check 1
   // Handle the extra thread case where the image width or height 
   // 
   if (x >= width || y >= height)
      return;

   // STUDENT: Check 2
   // Handle the border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[index] = g_DataIn[index];
       return;
    }

   // Perform the first load of values into shared memory
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
   sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();


   // STUDENT: Make sure only the thread ids should write the sum of the neighbors.
   float sumX = 0;
   for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++)
   {
       for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++)
       {
           PIXEL = (float) (g_DataIn[y*width + x + (dy*width + dx)]);
           sumX += PIXEL * HighBoostMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx + FILTER_RADIUS)];
       }
   }

   g_DataOut[index] = CLAMP_8bit((int)(PIXEL + HIGH_BOOST_FACTOR * (uint8_t)(PIXEL - sumX/9)));
}


#endif // _FILTER_KERNEL_H_


