#include <cub/cub.cuh>
#include <iostream>
#include "depthwise_common.h"
#include "int_divider.h"

template <typename T>
__global__ void __launch_bounds__(512, 2)
    DepthwiseConv2dBackwardFilterKernel(const DepthwiseArgs args,
                                        const T *__restrict__ out_backprop,
                                        const T *__restrict__ input,
                                        T *__restrict__ filter_backprop) {
  const int batch_num = args.batch;
  const int in_depth = args.in_depth;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int filter_width = args.filter_cols;
  const int stride_height = args.stride;
  const int stride_width = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_depth = args.out_depth;
  const int out_height = args.out_rows;
#ifdef USE_FAST_INTDIV
  const FastDividerUint32 out_width = args.out_cols;
  const FastDividerUint32 depth_multiplier = args.depth_multiplier;
#else
  const int out_width = args.out_cols;
  const int depth_multiplier = args.depth_multiplier;
#endif
  assert(gridDim.x == filter_width);
  assert(gridDim.z == out_depth);

  typedef cub::WarpReduce<T> WarpReduce;
  typename WarpReduce::TempStorage temp_storage;

  const int filter_w = blockIdx.x;
  const int filter_h = blockIdx.y;
  const int out_c = blockIdx.z;

  const int in_c = out_c / depth_multiplier;
  const int dm = out_c % depth_multiplier;
  const int filter_backprop_offset =
      (((filter_h * filter_width) + filter_w) * in_depth + in_c) *
          depth_multiplier +
      dm;
  const int out_spatial_size = out_height * out_width;

  T partial_sum = 0.;
  for (int batch = 0; batch < batch_num; batch++) {
    const int input_offset_temp = (batch * in_depth + in_c) * in_height;
    const int output_backprop_offset_temp =
        (batch * out_depth + out_c) * out_height;
    for (int i = threadIdx.x; i < out_spatial_size; i += blockDim.x) {
      const int out_col = i % out_width;
      const int out_row = i / out_width;
      // We use the formula: `(in_row - filter_w + pad_left ) / stride =
      // out_row` to compute corresponding in_row and out_row positions. Similar
      // for in_col and out_col.
      const int in_row = out_row * stride_height + filter_h - pad_height;
      const int in_col = out_col * stride_width + filter_w - pad_width;

      if (in_row < 0 || in_col < 0 || in_row >= in_height ||
          in_col >= in_width) {
        continue;
      }

      int input_offset = (input_offset_temp + in_row) * in_width + in_col;
      int output_backprop_offset =
          (output_backprop_offset_temp + out_row) * out_width + out_col;
      partial_sum += out_backprop[output_backprop_offset] * input[input_offset];
    }
  }

  T val = WarpReduce(temp_storage).Sum(partial_sum);
  if (cub::LaneId() == 0) {
    T *addr = filter_backprop + filter_backprop_offset;
    GpuAtomicAdd(addr, val);
  }
}

int main(int argc, char **argv) {
  DepthwiseArgs args;
  set_up(argc, argv, args);

  int num_out_backprop =
      args.batch * args.out_depth * args.out_rows * args.out_cols;
  int num_input = args.batch * args.in_depth * args.in_rows * args.in_cols;
  int num_filter_backprop = args.in_depth * args.filter_rows *
                            args.filter_cols * args.depth_multiplier;

  int out_backprop_bytes = sizeof(float) * num_out_backprop;
  int input_bytes = sizeof(float) * num_input;
  int filter_backprop_bytes = sizeof(float) * num_filter_backprop;

  float *out_backprop;
  float *input;
  float *filter_backprop;
  checkCUDA(cudaMalloc(&out_backprop, out_backprop_bytes));
  checkCUDA(cudaMalloc(&input, input_bytes));
  checkCUDA(cudaMalloc(&filter_backprop, filter_backprop_bytes));

  init_array(out_backprop, num_out_backprop);
  init_array(input, num_input);
  // TODO really need this?
  init_array(filter_backprop, num_filter_backprop, 0.0);

  auto device_fn = DepthwiseConv2dBackwardFilterKernel<float>;
  dim3 blocks = dim3(args.filter_cols, args.filter_rows, args.out_depth);
  dim3 threads = dim3(512, 1, 1);
  printf(">>> LOG: blocks: %d %d %d, threads: %d %d %d\n",
         blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

  auto launcher = [&](int repeats) {
    for (int i = 0; i < repeats; i++) {
      device_fn<<<blocks, threads>>>(args, out_backprop, input,
                                     filter_backprop);
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

  print_array(filter_backprop, num_filter_backprop, "Results:");
}
