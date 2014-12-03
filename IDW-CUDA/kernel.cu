
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_functions_decls.h"
#include "math_functions.h"

#include <stdio.h>

#include <Windows.h>
#include <time.h>

//#include <math.h>
//#define BENCHMARK
#define N_TESTS 1
#define IMG_2D_POS make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
#define SHARED_MEM_SIZE 49152
#define TILE_SIZE 32
#define ERROR_TOLERANCE 0.001


const int tileSize2d = TILE_SIZE * TILE_SIZE;

cudaError_t checkCudaError(cudaError_t status, const char* cmd, const char* file, int line) 
{
  if (status != cudaSuccess){
    printf("CUDA error %s, at %s:%i\n", status, cmd, line);
  }
  return status;
}
#ifdef __DEBUG
#define CUDA_CHECK(cmd) checkCudaError(cmd, #cmd, __FILE__, __LINE__);

#else 
#define CUDA_CHECK(cmd) cmd
#endif
void IDWCPU(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints);
cudaError_t IDWCuda(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints);
cudaError_t IDWCudaSH(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints);

__device__ __forceinline__ void calcIDWTile(int row, int col, float power, float &idwN, float &idwD, float &singular, float &singularValue, float3 *sp)
{  
  int i = 0;
  for (i = 0; i < tileSize2d; ++i) {
    float w = 0;
    
    float3 p = sp[i];       
    float d = fabsf(p.x - col + p.y - row); // Manhattan distance    
    
    if (d < 0.000001f) { // Avoid division by 0
      if (!singular){ 
        singular = 1;
        singularValue = p.z;
      }
      break;
    }
    else {
      w = __powf(d, -power);
      idwN += w * p.z;
      idwD += w;
    }
  }//end for   
}

__device__ __forceinline__ float calcIDW(int row, int col, float3 const* __restrict__ points, int nPoints, float power)
{
  float dfNominator = 0, dfDenominator = 0;
  int i = 0;
  for (i = 0; i < nPoints; ++i) {
    float w = 0;
    float3 p = points[i];
    float d = fabsf(p.x - col + p.y - row); // Manhattan distance   
    //d = d < 0 ? -d : d;
    if (d < 0.000001f) { // Avoid division by 0      
      break;
    }
    else {
      w = __powf(d, -power);
      dfNominator += w * p.z;
      dfDenominator += w;
    }
  }//end for

  if (i != nPoints) {
    return points[i].z;
  }
  else
    return dfNominator / dfDenominator;
}

__host__ __forceinline__ float calcIDWCPU(int row, int col, float3 const* __restrict__ points, int nPoints, float power)
{
  float dfNominator = 0, dfDenominator = 0;
  int i = 0;
  for (i = 0; i < nPoints; ++i) {
    float w = 0;
    float3 p = points[i];
    float d = fabsf(p.x - col + p.y - row); // Manhattan distance   
    //d = d < 0 ? -d : d;
    if (d < 0.000001f) { // Avoid division by 0      
      break;
    }
    else {
      w = 1.0f / powf(d, power);
      dfNominator += w * p.z;
      dfDenominator += w;
    }
  }//end for

  if (i != nPoints) {
    return points[i].z;
  }
  else
    return dfNominator / dfDenominator;
}

extern "C"
__global__ void interpolate(float *d_data, float power, int height, int width, float3 const* __restrict__ d_points, int nPoints) 
{
  const int2 thread_2D_img_pos = IMG_2D_POS;    

  const int thread_1D_img_pos = thread_2D_img_pos.y * width
    + thread_2D_img_pos.x;

  if (thread_2D_img_pos.x >= width || thread_2D_img_pos.y > height)
    return;

  d_data[thread_1D_img_pos] = calcIDW(thread_2D_img_pos.y, thread_2D_img_pos.x, d_points, nPoints, power);

}


extern "C"
__global__ void interpolateShared(float *d_data, float power, int height, int width, float3 const* __restrict__ d_points, int nPoints)
{
  const int2 pos2D = IMG_2D_POS;

  __shared__ float3 sharedPoints[TILE_SIZE * TILE_SIZE];

  if (pos2D.x >= width || pos2D.y > height)
    return;

  //int tidxSH = threadIdx.y * blockDim.x + threadIdx.x;
  //int tidxSH = threadIdx.y * blockDim.x + threadIdx.x;

  float idwN = 0.0, idwD = 0.0;  
  float singular = 0; //singular case where d(x, y) == 0;
  float singularValue = -1.0;

  //float idwN2 = 0.0, idwD2 = 0.0;
  //float singular2 = 0; //singular case where d(x, y) == 0;
  //float singularValue2 = -1.0;
  
  int i, tile = 0;
  
  for (i = 0, tile = 0; i < nPoints; i += tileSize2d, ++tile) {  
    
    sharedPoints[threadIdx.y * blockDim.x + threadIdx.x] = d_points[tile * tileSize2d + (threadIdx.y * blockDim.x + threadIdx.x)];
    //sharedPoints[(threadIdx.y + blockDim.y) * blockDim.x + threadIdx.x] = d_points[tile * tileSize2d + ((threadIdx.y + blockDim.y) * blockDim.x + threadIdx.x)];
    __syncthreads();//---------------------------

    calcIDWTile(pos2D.y, pos2D.x, power, idwN, idwD, singular, singularValue, sharedPoints);
    //calcIDWTile(pos2D.y+blockDim.y, pos2D.x, power, idwN2, idwD2, singular2, singularValue2, sharedPoints);

    __syncthreads();//---------------------------
  }
  
  if (singular) {
    d_data[pos2D.y * width + pos2D.x] = singularValue;
  }else { 
    d_data[pos2D.y * width + pos2D.x] = idwN / idwD;
  }
  
  /*if (singular2) {
    d_data[(pos2D.y + blockDim.y) * width + pos2D.x] = singularValue2;
  }
  else {
    d_data[(pos2D.y + blockDim.y) * width + pos2D.x] = idwN2 / idwD2;
  }*/
}

void verifyResults(float *control, float *ret, int h, int w)
{
  ////Verify results:
  bool ok = true;
  for (int i = 0; i < h; i++)
  {
    //printf("%d: ", i);
    for (int j = 0; j < w; j++)
    {
      //printf("data[%d][%d]:%.3f ",i,j, ret_data[i*h + w]);
      //printf("%.3f ", ret_data[i*w + j]);
      int currPos = i*w + j;

      if (abs(control[currPos] - ret[currPos]) > ERROR_TOLERANCE) {
        //printf("\n-------\nAssertError at pos%d, expected%.3f, encountered%.3f\n--------\n", currPos, control[currPos], ret_data[currPos]);
        ok = false;
      }
    }
    //printf("\n");
  }
  if (!ok) {
    printf("Results not ok. Tolerance:%.6f\n", ERROR_TOLERANCE);
  }
  /*else {
    printf("Error Tolerance OK!(%.6f)\n", ERROR_TOLERANCE);
  }*/
}

int main()
{  
  const int w = 512;
  const int h = 512;
  
  clock_t end, start;

  cudaError_t cudaStatus;
  
  //for (int i = 1; i < 200; i += 10) { 
  const int n_knowPoints = 1 * 1024;
  float3 *knowPoints = (float3 *) malloc(sizeof(float3) * n_knowPoints);

  float percent = (float)n_knowPoints / (w * h );
  for (int i = 0; i < n_knowPoints; i++)
  {
    float3 p = make_float3((int)((float)rand() / RAND_MAX * w), (int)((float)rand() / RAND_MAX * h), (float)rand() / RAND_MAX * 100);
    knowPoints[i] = p;
  }

  printf("Test: percent[%.3f], w %d, h %d----------------------\n", percent, w, h);
  float* ret_data;
  CUDA_CHECK(cudaMallocHost(&ret_data, sizeof(float) * h * w));
  
  ////IDW on CPU
  float* control;
  CUDA_CHECK(cudaMallocHost(&control, sizeof(float) * h * w));
  
  start = clock();
  IDWCPU(control, knowPoints, w, h, 2.0f, n_knowPoints);
  end = clock();
  printf("IDWCPU WALL Time: %.5fms\n\n\n", 1000.0f * (float)(end - start) / CLOCKS_PER_SEC);

  //// IDW in parallel.
  start = clock();
  cudaStatus = IDWCuda(ret_data, knowPoints, w, h, 2.0f, n_knowPoints);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Cuda failed!");
    return 1;
  }
  end = clock();
  printf("IDWCuda WALL Time: %.5fms\n\n\n", 1000.0f * (float)(end - start) / CLOCKS_PER_SEC);

  verifyResults(control, ret_data, h, w);

  start = clock();
  //// IDW in parallel using shared memory.
  cudaStatus = IDWCudaSH(ret_data, knowPoints, w, h, 2.0f, n_knowPoints);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Cuda failed!");
    return 1;
  }
  end = clock();
  printf("IDWCudaSH WALL Time: %.5fms\n\n\n", 1000.0f * (float)(end - start) / CLOCKS_PER_SEC);

  verifyResults(control, ret_data, h, w);
   
  
  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  //system("pause");
  return 0;
}


cudaError_t IDWCuda(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints)
{
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  

  float *dev_data;
  float3 *dev_knownPoints;
  cudaError_t cudaStatus;

  //Malloc
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_data, sizeof(float) * w * h));
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_knownPoints, sizeof(float3) * n_knownPoints));
  
  //Transfer HOST to DEVICE  
  CUDA_CHECK(cudaStatus = cudaMemcpy(dev_knownPoints, knownPoints, sizeof(float3) * n_knownPoints, cudaMemcpyHostToDevice));

  dim3 block = dim3(TILE_SIZE, TILE_SIZE, 1);
  dim3 grid = dim3((w / TILE_SIZE), (h / TILE_SIZE), 1);
  
  
  //------------------
  // --USING GLOBAL MEMORY
  //---------------------
  cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1);
  

  
  CUDA_CHECK(cudaEventRecord(start));
  //interpolate(float *d_data, int power, int height, int width, int3 *d_points, int nPoints)
  for (int i = 0; i < N_TESTS; i++)
  {
    interpolate << <grid, block >> >(dev_data, power, h, w, dev_knownPoints, n_knownPoints);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaStatus = cudaDeviceSynchronize());
  CUDA_CHECK(cudaStatus = cudaGetLastError());

  //CopyBack
  CUDA_CHECK(cudaStatus = cudaMemcpy(ret_data, dev_data, sizeof(float) * w * h, cudaMemcpyDeviceToHost));

  
  float msecTotal;
  CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));
  printf("msecTimeCUDA: %.5fms\n", msecTotal / N_TESTS);

  CUDA_CHECK(cudaFree(dev_data));
  CUDA_CHECK(cudaFree(dev_knownPoints));

  return cudaStatus;
}

cudaError_t IDWCudaSH(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints)
{
  float *dev_data;
  float3 *dev_knownPoints;
  cudaError_t cudaStatus;


  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  
  //Malloc
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_data, sizeof(float) * w * h));
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_knownPoints, sizeof(float3) * n_knownPoints));

  //Transfer HOST to DEVICE  
  CUDA_CHECK(cudaStatus = cudaMemcpy(dev_knownPoints, knownPoints, sizeof(float3) * n_knownPoints, cudaMemcpyHostToDevice));

  dim3 block = dim3(TILE_SIZE, TILE_SIZE, 1);
  dim3 grid = dim3((w / TILE_SIZE), (h / TILE_SIZE), 1);
    

  //---------------------
  // --USING SHARED MEMORY
  //---------------------
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferShared);
  
  CUDA_CHECK(cudaEventRecord(start));
  //interpolate(float *d_data, int power, int height, int width, int3 *d_points, int nPoints) 
  //interpolateShared << <grid, block >> >(dev_data, power, h, w, dev_knownPoints, n_knownPoints);
  for (int i = 0; i < N_TESTS; i++)
  {
    interpolateShared << <grid, block >> >(dev_data, power, h, w, dev_knownPoints, n_knownPoints);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaStatus = cudaDeviceSynchronize());

  CUDA_CHECK(cudaStatus = cudaGetLastError());


  //CopyBack
  CUDA_CHECK(cudaStatus = cudaMemcpy(ret_data, dev_data, sizeof(float) * w * h, cudaMemcpyDeviceToHost));

  
  float msecTotal;
  CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("msecTimeCUDAShared: %.5fms\n", msecTotal / N_TESTS);

  CUDA_CHECK(cudaFree(dev_data));
  CUDA_CHECK(cudaFree(dev_knownPoints));

  return cudaStatus;
}

void IDWCPU(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints)
{
  int nOP = w * h;

  for (int i = 0; i < nOP; i++)
  {
    ret_data[i] = calcIDWCPU(i / w, i % w, knownPoints, n_knownPoints, power);
  }
}