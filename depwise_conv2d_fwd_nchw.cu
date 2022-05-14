#include <cub/cub.cuh>
#include <iostream>
#include "depthwise_common.h"
#include "int_divider.h"

// A GPU kernel to compute the depthwise convolution forward pass
// in NCHW format.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight,
          int kKnownDepthMultiplier>
__global__ void __launch_bounds__(1024, 2)
    DepthwiseConv2dGPUKernelNCHW(const DepthwiseArgs args,
                                 const T* __restrict__ input,
                                 const T* __restrict__ filter,
                                 T* __restrict__ output, int num_outputs) {
  // typedef typename detail::PseudoHalfType<T>::Type S;
  using S = float;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_height =
      kKnownFilterHeight < 0 ? args.filter_rows : kKnownFilterHeight;
  const int filter_width =
      kKnownFilterWidth < 0 ? args.filter_cols : kKnownFilterWidth;
#ifdef USE_FAST_INTDIV
  const FastDividerUint32 depth_multiplier(kKnownDepthMultiplier < 0 ?
      args.depth_multiplier : kKnownDepthMultiplier);
#else
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
#endif
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
#ifdef USE_FAST_INTDIV
  const FastDividerUint32 out_depth(args.out_depth);
  const FastDividerUint32 out_height(args.out_rows);
#else
  const int out_depth = args.out_depth;
  const int out_height = args.out_rows;
#endif
  const int out_width = args.out_cols;

  GPU_1D_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_col = thread_id % out_width;
    const int out_row = (thread_id / out_width) % out_height;
    const int out_channel = (thread_id / out_width / out_height) % out_depth;
    const int batch = thread_id / out_width / out_height / out_depth;

    // Compute the input depth and the index of depth multiplier
    // based off the output depth index that this thread is
    // computing n.
    const int in_channel = out_channel / depth_multiplier;
    const int multiplier = out_channel % depth_multiplier;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_depth * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp =
        (batch * in_depth + in_channel) * (in_height * in_width);

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_row_start = out_row * stride - pad_height;
    const int input_col_start = out_col * stride - pad_width;
    const int input_row_end = input_row_start + filter_height;
    const int input_col_end = input_col_start + filter_width;

    S sum = static_cast<S>(0);
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_height && input_col_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = input_col_start + filter_col;

          const int input_offset =
              (input_offset_temp) + (in_row * in_width) + in_col;
          const int filter_offset =
              multiplier +
              depth_multiplier *
                  (in_channel + in_depth * (filter_col + filter_offset_temp));
          sum += static_cast<S>(ldg(input + input_offset)) *
                 static_cast<S>(ldg(filter + filter_offset));
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      UNROLL for (int filter_row = 0; filter_row < filter_height;
                  ++filter_row) {
        const int in_row = input_row_start + filter_row;
        const int filter_offset_temp = filter_width * filter_row;
        UNROLL for (int filter_col = 0; filter_col < filter_width;
                    ++filter_col) {
          const int in_col = input_col_start + filter_col;
          // TODO(vrv): the in_row check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int in_col = input_col_start + filter_col;

            // input_offset_temp indexes into the start of memory
            // where the spatial data starts.
            const int input_offset =
                (input_offset_temp) + (in_row * in_width) + in_col;

            const int filter_offset =
                multiplier +
                depth_multiplier *
                    (in_channel + in_depth * (filter_col + filter_offset_temp));
            sum += static_cast<S>(ldg(input + input_offset)) *
                   static_cast<S>(ldg(filter + filter_offset));
          }
        }
      }
    }

    output[thread_id] = static_cast<T>(sum);
  }
}

int main(int argc, char **argv) {
  DepthwiseArgs args;
  set_up(argc, argv, args);
  
  int num_input = args.batch * args.in_depth * args.in_rows * args.in_cols;
  int num_filter= args.in_depth * args.filter_rows * args.filter_cols *
                  args.depth_multiplier;
  int num_output = args.batch * args.out_depth * args.out_rows * args.out_cols;

  int output_bytes = sizeof(float) * num_output;
  int input_bytes = sizeof(float) * num_input;
  int filter_bytes = sizeof(float) * num_filter;

  float *input;
  float *filter;
  float *output;
  checkCUDA(cudaMalloc(&input, input_bytes));
  checkCUDA(cudaMalloc(&filter, filter_bytes));
  checkCUDA(cudaMalloc(&output, output_bytes));

  init_array(input, num_input);
  init_array(filter, num_filter);
  // TODO really need this?
  init_array(output, num_output, 0.0);

  auto device_fn = DepthwiseConv2dGPUKernelNCHW<float, -1, -1, -1>;
  int blocks = 246;
  int threads = 512;
  checkCUDA(
      cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, device_fn, 0, 0));
  const int max_block_count = std::numeric_limits<int>::max();
  blocks = std::min(max_block_count, blocks);
  printf(">>> LOG: blocks: %d, threads: %d\n", blocks, threads);

  auto launcher = [&](int repeats) {
    for (int i = 0; i < repeats; i++) {
      device_fn<<<blocks, threads>>>(args, input, filter, output, num_output);
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

  print_array(output, num_output, "Results:");
}

