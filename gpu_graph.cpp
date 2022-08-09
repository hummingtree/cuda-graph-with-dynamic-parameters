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
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

gpu_graph_t::~gpu_graph_t() {
  if (_instantiated) {
    cudaErrCheck(cudaGraphDestroy(_graph));
    cudaErrCheck(cudaGraphExecDestroy(_graph_exec));
    _instantiated = false;
  }
}

void gpu_graph_t::begin_capture(cudaStream_t s) {
  cudaErrCheck(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
}

void gpu_graph_t::end_capture(cudaStream_t s) {
  if (_instantiated) { cudaErrCheck(cudaGraphDestroy(_graph)); }
  cudaErrCheck(cudaStreamEndCapture(s, &_graph));

  bool need_instantiation;

  if (_instantiated) {
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;
    // First we try to update the graph as this is much cheaper than re-instantiation
    cudaErrCheck(cudaGraphExecUpdate(_graph_exec, _graph, &errorNode, &updateResult));
    if (_graph_exec == nullptr || updateResult != cudaGraphExecUpdateSuccess) {
      // The update is unsuccessful, need to re-instantiate
      cudaGetLastError(); // <- Clear the error state
      if (_graph_exec != nullptr) { cudaErrCheck(cudaGraphExecDestroy(_graph_exec)); }
      need_instantiation = true;
    } else {
      // The update is successful, no need to re-instantiate
      need_instantiation = false;
    }
  } else {
    need_instantiation = true;
  }

  if (need_instantiation) {
    cudaErrCheck(cudaGraphInstantiate(&_graph_exec, _graph, nullptr, nullptr, 0));
  }

  _instantiated = true;
}

void gpu_graph_t::launch_graph(cudaStream_t s) {
  if (_instantiated) {
    cudaErrCheck(cudaGraphLaunch(_graph_exec, s));
  } else {
    fprintf(stderr, "Launching an invalid or uninstantiated graph\n");
  }
}

void gpu_graph_t::add_kernel_node(size_t key, cudaKernelNodeParams params, cudaStream_t stream)
{
  // Get the currently capturing graph
  cudaStreamCaptureStatus capture_status;
  cudaGraph_t graph;
  const cudaGraphNode_t *deps;
  size_t dep_count;
  cudaErrCheck(cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &graph, &deps, &dep_count));

  // Now add a new node
  cudaGraphNode_t new_node;
  cudaErrCheck(cudaGraphAddKernelNode(&new_node, graph, deps, dep_count, &params));
  _node_map[key] = new_node;
  // Update the stream dependency
  cudaErrCheck(cudaStreamUpdateCaptureDependencies(stream, &new_node, 1, 1));
}

void gpu_graph_t::update_kernel_node(size_t key, cudaKernelNodeParams params)
{
  cudaErrCheck(cudaGraphExecKernelNodeSetParams(_graph_exec, _node_map[key], &params));
}
