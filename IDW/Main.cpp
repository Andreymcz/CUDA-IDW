#include <stdio.h>

#include <Windows.h>
#include <time.h>

inline float calcIDWCPU(int row, int col, float3 *points, int nPoints)
{
  float dfNominator = 0, dfDenominator = 0, dfR2 = 0, dfRX = 0, dfRY = 0, invdist = 0;
  int i = 0;
  for (i = 0; i < nPoints; ++i) {


    dfRX = points[i].x - col;
    dfRY = points[i].y - row;
    dfR2 = dfRX * dfRX + dfRY * dfRY;

    //dfR2 = (points[i].x - col) * (points[i].x - col)
    //		+ (points[i].y - row) * (points[i].y - row);
    // If the test point is close to the grid node, use the point
    // value directly as a node value to avoid singularity.
    if (dfR2 < 0.001f) {
      break;
    }
    else {
      invdist = 1.f / (dfR2 * dfR2);
      dfNominator += invdist * points[i].z;
      dfDenominator += invdist;
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

int main()
{
  //const int arraySize = 5;
  //const int a[arraySize] = { 1, 2, 3, 4, 5 };
  //const int b[arraySize] = { 10, 20, 30, 40, 50 };
  //int c[arraySize] = { 0 };

  const int w = 1000;
  const int h = 1000;
  const int n_knowPoints = 10;
  float3 *knowPoints = (float3 *)malloc(sizeof(float3) * n_knowPoints);

  for (int i = 0; i < n_knowPoints; i++)
  {
    knowPoints[i] = make_float3((float)rand() / RAND_MAX * w, (float)rand() / RAND_MAX * h, (float)rand() / RAND_MAX * 100);
  }

  float* ret_data;
  CUDA_CHECK(cudaMallocHost(&ret_data, sizeof(float) * h * w));

  // IDW in parallel.
  cudaError_t cudaStatus = IDWCuda(ret_data, knowPoints, w, h, 2, n_knowPoints);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "addWithCuda failed!");
    return 1;
  }


  //IDW on CPU
  IDWCPU(ret_data, knowPoints, w, h, 2, n_knowPoints);


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

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  system("pause");
  return 0;
}

cudaError_t IDWCuda(float *ret_data, float3 *knownPoints, int w, int h, int power, int n_knownPoints)
{
  float *dev_data;
  float3 *dev_knownPoints;
  cudaError_t cudaStatus;

  //Malloc
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_data, sizeof(float) * w * h));
  CUDA_CHECK(cudaStatus = cudaMalloc((void**)&dev_knownPoints, sizeof(float3) * n_knownPoints));

  //Transfer HOST to DEVICE  
  CUDA_CHECK(cudaStatus = cudaMemcpy(dev_knownPoints, knownPoints, sizeof(float3) * n_knownPoints, cudaMemcpyHostToDevice));

  dim3 block = dim3(16, 16, 1);
  dim3 grid = dim3(1 + (w / block.x), 1 + (h / block.y), 1);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  //interpolate(float *d_data, int power, int height, int width, int3 *d_points, int nPoints) 
  interpolate << <grid, block >> >(dev_data, power, h, w, dev_knownPoints, n_knownPoints);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaStatus = cudaDeviceSynchronize());

  CUDA_CHECK(cudaStatus = cudaGetLastError());

  float msecTotal = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

  printf("msecTimeCUDA: %.5fms\n", msecTotal);

  //CopyBack
  CUDA_CHECK(cudaStatus = cudaMemcpy(ret_data, dev_data, sizeof(float) * w * h, cudaMemcpyDeviceToHost));

  return cudaStatus;
}

void IDWCPU(float *ret_data, float3 *knownPoints, int w, int h, int power, int n_knownPoints)
{
  clock_t start = clock();

  int nOP = w * h;

  for (int i = 0; i < nOP; i++)
  {
    ret_data[i] = calcIDWCPU(i / w, i % w, knownPoints, n_knownPoints);
  }

  clock_t end = clock();
  printf("CPU Time: %.5fms\n", 1000.0f * (float)(end - start) / CLOCKS_PER_SEC);
}