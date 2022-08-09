A standalone example showing a strategy to allow modifying kernel parameters and launch configurations in a graph that is created and instantiated with the stream capture APIs. It utilizes the `cudaStreamGetCaptureInfo_v2` and `cudaStreamUpdateCaptureDependencies` that are instroduced in CUDA 11.3 toolkit.

This is one possible solution to the question "How does one use the stream capture to create a CUDA graph while still being able to change the parameters dynamically".
