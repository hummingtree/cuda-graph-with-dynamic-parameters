/**
 * Copyright (c) 2022, NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gpu_graph.hpp"
#include "cuda_helper.hpp"

constexpr int n_kernel = 2;
constexpr int n_iteration = 40000;

__global__ void shortKernel(float *out_d, const float *in_d, int N, float f){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) { 
      out_d[idx] = f * in_d[idx];
  }
}

__global__ void initKernel(float *ptr, int N, float f){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) { 
      ptr[idx] = f;
  }
}

void run_kernels_graph(float *out_d, float *in_d, int size, float f, gpu_graph_t &g, cudaStream_t s)
{
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;

  for(int i = 0; i < n_kernel; i++){
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(threads), 1, 1};
    params.gridDim = {static_cast<unsigned int>(blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void *>(shortKernel);
    void *args[] = {&out_d, &in_d, &size, &f};
    params.kernelParams = args;
    params.extra = nullptr;

    if (g.state() == gpu_graph_t::state_t::capture) {
      // Static kernels
      shortKernel<<<blocks, threads, 0, s>>>(out_d, in_d, size, 1.004f);
      shortKernel<<<blocks, threads, 0, s>>>(in_d, out_d, size, 1.004f);

      // kernels with dynamic parameter `f`
      // Add the kernel nodes
      g.add_kernel_node(i * 2 + 0, params, s);
      params.kernelParams[0] = &in_d;
      params.kernelParams[1] = &out_d;
      g.add_kernel_node(i * 2 + 1, params, s);
    } else if (g.state() == gpu_graph_t::state_t::update) {
      // Update the kernel nodes
      g.update_kernel_node(i * 2 + 0, params);
      params.kernelParams[0] = &in_d;
      params.kernelParams[1] = &out_d;
      g.update_kernel_node(i * 2 + 1, params);
    }
  } 
}

void run_kernels_no_graph(float *out_d, float *in_d, int size, float f, cudaStream_t s)
{
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;

  for(int i = 0; i < n_kernel; i++){
    // Static kernels
    shortKernel<<<blocks, threads, 0, s>>>(out_d, in_d, size, 1.004f);
    shortKernel<<<blocks, threads, 0, s>>>(in_d, out_d, size, 1.004f);

    // kernels with dynamic parameter `f`
    shortKernel<<<blocks, threads, 0, s>>>(out_d, in_d, size, f);
    shortKernel<<<blocks, threads, 0, s>>>(in_d, out_d, size, f);
  } 
}

void run_init(float *ptr, int size, float f, cudaStream_t s) {
  constexpr int threads = 256;
  int blocks = (size + threads - 1) / threads;
  initKernel<<<blocks, threads, 0, s>>>(ptr, size, f);
}

int main() 
{
  gpu_graph_t _graph;
  gpu_graph_t _graph_always_recapture;

  _graph_always_recapture._always_recapture = true;

  // Set up memory, stream, events
  float *out_d = nullptr;
  float *in_d = nullptr;

  int size = 32;
  size_t bytes = size * sizeof(float);

  cudaErrCheck(cudaMalloc(&out_d, bytes));
  cudaErrCheck(cudaMalloc(&in_d, bytes));

  cudaStream_t stream;
  cudaEvent_t start, stop;
  cudaErrCheck(cudaStreamCreate(&stream));
  cudaErrCheck(cudaEventCreate(&start));
  cudaErrCheck(cudaEventCreate(&stop));

  float scale = 1.0f;

  auto wrap_obj_graph = [&](gpu_graph_t &g, cudaStream_t s) {
    run_kernels_graph(out_d, in_d, size, scale, g, s);
  };

  auto wrap_obj_no_graph = [&](gpu_graph_t &g, cudaStream_t s) {
    run_kernels_no_graph(out_d, in_d, size, scale, stream);
  };

  run_init(out_d, size, 1.0f, stream);
  run_init(in_d, size, 1.0f, stream);

  printf("Running with    CUDA graph ('Recapture-then-update') ...\n");

  // Running the test with graph
  cudaErrCheck(cudaEventRecord(start, stream));

  for(int i = 0; i < n_iteration; i++){
    scale = 1.0f + i * 0.001f;
    _graph_always_recapture.wrap(wrap_obj_no_graph, stream);
  }

  cudaErrCheck(cudaEventRecord(stop, stream));
  cudaErrCheck(cudaEventSynchronize(stop));

  float milliseconds;
  cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Running with    CUDA graph ('Recapture-then-update') took %6.2f ms\n", milliseconds);

  run_init(out_d, size, 1.0f, stream);
  run_init(in_d, size, 1.0f, stream);

  printf("Running with    CUDA graph ('Combined Approach')     ...\n");

  // Running the test with graph
  cudaErrCheck(cudaEventRecord(start, stream));

  for(int i = 0; i < n_iteration; i++){
    scale = 1.0f + i * 0.001f;
    _graph.wrap(wrap_obj_graph, stream);
  }

  cudaErrCheck(cudaEventRecord(stop, stream));
  cudaErrCheck(cudaEventSynchronize(stop));

  cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Running with    CUDA graph ('Combined Approach')     took %6.2f ms\n", milliseconds);

  run_init(out_d, size, 1.0f, stream);
  run_init(in_d, size, 1.0f, stream);

  printf("Running without CUDA graph                           ...\n");

  cudaErrCheck(cudaEventRecord(start, stream));

  for(int i = 0; i < n_iteration; i++){
    scale = 1.0f + i * 0.001f;
    run_kernels_no_graph(out_d, in_d, size, scale, stream);
  }

  cudaErrCheck(cudaEventRecord(stop, stream));
  cudaErrCheck(cudaEventSynchronize(stop));

  cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Running without CUDA graph                           took %6.2f ms\n", milliseconds);

  // Finalize memory, stream, events
  cudaErrCheck(cudaStreamDestroy(stream));
  cudaErrCheck(cudaEventDestroy(start));
  cudaErrCheck(cudaEventDestroy(stop));

  cudaErrCheck(cudaFree(out_d));
  cudaErrCheck(cudaFree(in_d));
}
