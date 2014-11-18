
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <Windows.h>
#include <time.h>
#include <math.h>
#define BENCHMARK
#define N_TESTS 10
#define IMG_2D_POS make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

cudaError_t checkCudaError(cudaError_t status, const char* cmd, const char* file, int line) 
{
  if (status != cudaSuccess){
    printf("CUDA error %s, at %s:%i\n", status, cmd, line);
  }
  return status;
}
#define CUDA_CHECK(cmd) checkCudaError(cmd, #cmd, __FILE__, __LINE__);

void IDWCPU(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints);
cudaError_t IDWCuda(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints);

inline float calcIDWCPU(int row, int col, float3 *points, int nPoints, float power)
{
  float dfNominator = 0, dfDenominator = 0;
  int i = 0;
  for (i = 0; i < nPoints; ++i) {
    float w = 0;    
    float d = (points[i].x - col + points[i].y - row); // Manhattan distance   
       
    if (d < 0.000001f) { //Avoid division by 0      
      break;
    } 
    else {
      w = 1.f / pow(d, power);
      dfNominator += w * points[i].z;
      dfDenominator += w;
    }
  }

  if (i != nPoints) {
    return points[i].z;
  }
  else if (dfDenominator == 0.0) {
    return 0;
  }
  else
    return dfNominator / dfDenominator;
}

__device__ __forceinline__ float calcIDW(int row, int col, float3 const* __restrict__ points, int nPoints, float power)
{
  float dfNominator = 0, dfDenominator = 0;
  int i = 0;
  for (i = 0; i < nPoints; ++i) {
    float w = 0;
    float3 p = points[i];
    float d = (p.x - col + p.y - row); // Manhattan distance   

    if (d < 0.000001f) { // Avoid division by 0      
      break;
    }
    else {
      w = 1.f / pow(d, power);
      dfNominator += w * p.z;
      dfDenominator += w;
    }
  }//end for

  if (i != nPoints) {
    return points[i].z;
  }
  else if (dfDenominator == 0.0) {
    return 0;
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

int main()
{

  const int w = 1024;
  const int h = 1024;
  
  cudaError_t cudaStatus;
  float percent = 0.5;
  //for (float percent = 0.01; percent < 0.7; percent += 0.1) { 
    
    int n_knowPoints = w * h * percent; //10% known
    float3 *knowPoints = (float3 *) malloc(sizeof(float3) * n_knowPoints);

    for (int i = 0; i < n_knowPoints; i++)
    {
      knowPoints[i] = make_float3((float)rand() / RAND_MAX * w, (float)rand() / RAND_MAX * h, (float)rand() / RAND_MAX * 100);
    }

    printf("Test: percent[%.3f], w %d, h %d----------------------\n", percent, w, h);
    float* ret_data;
    CUDA_CHECK(cudaMallocHost(&ret_data, sizeof(float) * h * w));

      // IDW in parallel.
    cudaStatus = IDWCuda(ret_data, knowPoints, w, h, 2.0f, n_knowPoints);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "addWithCuda failed!");
      return 1;
    }

  
    //IDW on CPU
    IDWCPU(ret_data, knowPoints, w, h, 2.0f, n_knowPoints);
  

    //Print results:
    //for (int i = 0; i < h; i++)
    //{
    //  printf("%d: ", i);
    //  for (int j = 0; j < w; j++)
    //  {
    //    //printf("data[%d][%d]:%.3f ",i,j, ret_data[i*h + w]);
    //    printf("%.3f ", ret_data[i*w + j]);
    //  }
    //  printf("\n");
    //}
  //}
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
  float *dev_data;
  float3 *dev_knownPoints;
  cudaError_t cudaStatus;

  //Malloc
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_data, sizeof(float) * w * h));
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_knownPoints, sizeof(float3) * n_knownPoints));
  
  //Transfer HOST to DEVICE  
  CUDA_CHECK(cudaStatus = cudaMemcpy(dev_knownPoints, knownPoints, sizeof(float3) * n_knownPoints, cudaMemcpyHostToDevice));

  dim3 block = dim3(32, 32, 1);
  dim3 grid = dim3(1 + (w / block.x), 1 + (h / block.y), 1);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

#ifdef BENCHMARK
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < N_TESTS; i++)
  {
    //interpolate(float *d_data, int power, int height, int width, int3 *d_points, int nPoints) 
    interpolate << <grid, block >> >(dev_data, power, h, w, dev_knownPoints, n_knownPoints);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaStatus = cudaDeviceSynchronize());

  CUDA_CHECK(cudaStatus = cudaGetLastError());

  float msecTotal = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("msecTimeCUDA: %.5fms\n", msecTotal / N_TESTS);

#else  

  CUDA_CHECK(cudaEventRecord(start));

  //interpolate(float *d_data, int power, int height, int width, int3 *d_points, int nPoints) 
  interpolate << <grid, block >> >(dev_data, power, h, w, dev_knownPoints, n_knownPoints);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaStatus = cudaDeviceSynchronize());

  CUDA_CHECK(cudaStatus = cudaGetLastError());

  float msecTotal = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("msecTimeCUDA: %.5fms\n", msecTotal);

#endif // BENCKMARK



  //CopyBack
  CUDA_CHECK(cudaStatus = cudaMemcpy(ret_data, dev_data, sizeof(float) * w * h, cudaMemcpyDeviceToHost));

  return cudaStatus;
}

void IDWCPU(float *ret_data, float3 *knownPoints, int w, int h, float power, int n_knownPoints)
{
  clock_t start = clock();
#ifdef BENCHMARK

  for (int i = 0; i < N_TESTS; i++)
  {
    int nOP = w * h;
    for (int i = 0; i < nOP; i++)
    {
      ret_data[i] = calcIDWCPU(i / w, i % w, knownPoints, n_knownPoints, power);
    }
  }

  clock_t end = clock();
  printf("CPU Time: %.5fms\n", (1000.0f * (float)(end - start) / CLOCKS_PER_SEC) / N_TESTS);

#else
  int nOP = w * h;

  for (int i = 0; i < nOP; i++)
  {
    ret_data[i] = calcIDWCPU(i / w, i % w, knownPoints, n_knownPoints, power);
  }

  clock_t end = clock();
  printf("CPU Time: %.5fms\n", 1000.0f * (float)(end - start) / CLOCKS_PER_SEC);
#endif // BENCHMARK

  
}