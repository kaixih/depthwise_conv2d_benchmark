#include <cub/cub.cuh>
#include <iostream>
#include "depthwise_common.h"
#include "int_divider.h"

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(640, 2)
    DepthwiseConv2dBackpropInputGPUKernelNCHW(
        const DepthwiseArgs args, const T* __restrict__ out_backprop,
        const T* __restrict__ filter, T* __restrict__ in_backprop,
        int num_in_backprop) {
#ifdef USE_FAST_INTDIV
  const FastDividerUint32 in_height(args.in_rows);
  const FastDividerUint32 in_width(args.in_cols);
  const FastDividerUint32 in_depth(args.in_depth);
#else
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
#endif
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
#ifdef USE_FAST_INTDIV
  const FastDividerUint32 stride(args.stride);
#else
  const int stride = args.stride;
#endif
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

  // TODO(vrv): Consider assigning threads to output and using
  // atomics for accumulation, similar to the filter case.
  GPU_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    // Compute the indexes of this thread in the input.
    const int in_col = thread_id % in_width;
    const int in_row = (thread_id / in_width) % in_height;
    const int in_channel = (thread_id / in_width / in_height) % in_depth;
    const int batch = thread_id / in_depth / in_width / in_height;

    T sum = static_cast<T>(0);
    const int out_channel_start = in_channel * depth_multiplier;
    const int out_channel_end = out_channel_start + depth_multiplier;

    const int out_row_start =
        tf_max<int>(0, (in_row - filter_height + pad_height + stride) / stride);
    const int out_row_end =
        tf_min<int>(out_height - 1, (in_row + pad_height) / stride);
    const int out_col_start =
        tf_max<int>(0, (in_col - filter_width + pad_width + stride) / stride);
    const int out_col_end =
        tf_min<int>(out_width - 1, (in_col + pad_width) / stride);

    UNROLL for (int out_channel = out_channel_start;
                out_channel < out_channel_end; ++out_channel) {
      UNROLL for (int out_row = out_row_start; out_row <= out_row_end;
                  ++out_row) {
        const int filter_row = in_row + pad_height - out_row * stride;
        const int filter_dm = out_channel - out_channel_start;

        const int temp_filter_offset = filter_width * filter_row;
        for (int out_col = out_col_start; out_col <= out_col_end; ++out_col) {
          const int filter_col = in_col + pad_width - out_col * stride;
          const int filter_offset =
              filter_dm +
              args.depth_multiplier *
                  (in_channel + in_depth * (filter_col + temp_filter_offset));

          const int out_backprop_offset =
              (batch * out_depth * out_height * out_width) +
              (out_channel * out_height * out_width) + (out_row * out_width) +
              (out_col);

          sum += ldg(out_backprop + out_backprop_offset) *
                 ldg(filter + filter_offset);
        }
      }
    }
    const int in_backprop_offset = (batch * in_height * in_width * in_depth) +
                                   (in_channel * in_height * in_width) +
                                   (in_row * in_width) + (in_col);
    in_backprop[in_backprop_offset] = sum;
  }
}

int main(int argc, char **argv) {
  DepthwiseArgs args;
  set_up(argc, argv, args);
  
  int num_filter= args.in_depth * args.filter_rows * args.filter_cols *
                  args.depth_multiplier;
  int num_out_backprop = args.batch * args.out_depth * args.out_rows *
                         args.out_cols;
  int num_in_backprop = args.batch * args.in_depth * args.in_rows *
                        args.in_cols;

  int filter_bytes = sizeof(float) * num_filter;
  int out_backprop_bytes = sizeof(float) * num_out_backprop;
  int in_backprop_bytes = sizeof(float) * num_in_backprop;

  float *filter;
  float *out_backprop;
  float *in_backprop;
  checkCUDA(cudaMalloc(&filter, filter_bytes));
  checkCUDA(cudaMalloc(&out_backprop, out_backprop_bytes));
  checkCUDA(cudaMalloc(&in_backprop, in_backprop_bytes));

  init_array(filter, num_filter);
  init_array(out_backprop, num_out_backprop);
  // TODO really need this?
  init_array(in_backprop, num_in_backprop, 0.0);

  auto device_fn = DepthwiseConv2dBackpropInputGPUKernelNCHW<float, -1, -1, -1>;
  int blocks = 246;
  int threads = 512;
  checkCUDA(
      cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, device_fn, 0, 640));
  blocks = std::min(blocks, DivUp(num_in_backprop, threads));
  printf(">>> LOG: blocks: %d, threads: %d\n", blocks, threads);

  auto launcher = [&](int repeats) {
    for (int i = 0; i < repeats; i++) {
      device_fn<<<blocks, threads>>>(args, out_backprop, filter, in_backprop,
                                     num_in_backprop);
    }
  };
  // warmup
  launcher(20);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  const int repeats = 50;
  launcher(repeats);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf(">>> LOG: time(ms): %f\n", milliseconds / repeats);

  print_array(in_backprop, num_in_backprop, "Results:");
}


