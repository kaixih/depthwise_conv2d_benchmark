#include <cub/cub.cuh>
#include <iostream>
#include "int_divider.h"

#define checkCUDA(expression)                               \
  {                                                         \
    cudaError_t status = (expression);                      \
    if (status != cudaSuccess) {                            \
      std::cerr << "Error on line " << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                              \
    }                                                       \
  }

struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_rows;
  int in_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int depth_multiplier;
  int stride;
  int pad_rows;  // Amount of padding to the top of the input
  int pad_cols;  // Amount of padding to the left of the input

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  DepthwiseArgs()
      : batch(0),
        in_rows(0),
        in_cols(0),
        in_depth(0),
        filter_rows(0),
        filter_cols(0),
        depth_multiplier(0),
        stride(0),
        pad_rows(0),
        pad_cols(0),
        out_rows(0),
        out_cols(0),
        out_depth(0) {}
};

template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator &operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator &other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end)
      : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

template <typename T>
__device__ GpuGridRange<T> GpuGridRangeX(T count) {
  return GpuGridRange<T>(
      /*begin=*/blockIdx.x * blockDim.x + threadIdx.x,
      /*delta=*/gridDim.x * blockDim.x, /*end=*/count);
}

template <typename T>
__host__ __device__ T GpuLdg(const T *address) {
  return __ldg(address);
}

template <typename T>
__host__ __device__ inline T ldg(const T *ptr) {
  return GpuLdg(ptr);
}

#define GPU_1D_KERNEL_LOOP(i, n) for (int i : GpuGridRangeX<int>(n))
#define UNROLL _Pragma("unroll")
#define NOUNROLL _Pragma("nounroll")

template <typename T>
struct CudaSupportedTypeImpl {
  using type = T;
};

template <typename T>
using CudaSupportedType = typename CudaSupportedTypeImpl<T>::type;

template <typename T>
__device__ CudaSupportedType<T> *ToCudaSupportedPtr(T *ptr) {
  return reinterpret_cast<CudaSupportedType<T> *>(ptr);
}

template <typename From, typename To>
using ToTypeIfConvertible =
    typename std::enable_if<std::is_convertible<From, To>::value, To>::type;

template <typename T, typename U>
__device__ ToTypeIfConvertible<U, T> GpuAtomicAdd(T *ptr, U value) {
  return atomicAdd(ToCudaSupportedPtr(ptr), value);
  // return *ptr = value;
}



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
  const int depth_multiplier =
      kKnownDepthMultiplier < 0 ? args.depth_multiplier : kKnownDepthMultiplier;
  const int stride = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;
  const int out_depth = args.out_depth;

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

template <typename T>
void init_array(T *dev_ptr, int n) {
  T *host_ptr = new T[n];
  for (int i = 0; i < n; i++) {
    host_ptr[i] = 0.1;
  }
  checkCUDA(
      cudaMemcpy(dev_ptr, host_ptr, sizeof(T) * n, cudaMemcpyHostToDevice));
  delete[] host_ptr;
}

template <typename T>
void init_array(T *dev_ptr, int n, float v) {
  T *host_ptr = new T[n];
  for (int i = 0; i < n; i++) {
    host_ptr[i] = v;
  }
  checkCUDA(
      cudaMemcpy(dev_ptr, host_ptr, sizeof(T) * n, cudaMemcpyHostToDevice));
  delete[] host_ptr;
}

template <typename T>
void print_array(T *dev_ptr, int n, const std::string &prompt) {
  std::cout << prompt << std::endl;
  T *host_ptr = new T[n];
  checkCUDA(
      cudaMemcpy(host_ptr, dev_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    printf("%f, ", static_cast<float>(host_ptr[i]));
    if ((i + 1) % 10 == 0) break;
  }
  printf("\n");
  delete[] host_ptr;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char **argv) {
  int dargs[13] = {3, 128, 128, 144, 3, 3, 1, 1, 1, 1, 128, 128, 144};

  if (argc > 6) {
    dargs[0] = atoi(argv[1]);
    dargs[1] = atoi(argv[2]);
    dargs[2] = atoi(argv[3]);
    dargs[3] = atoi(argv[4]);
    dargs[4] = atoi(argv[5]);
    dargs[5] = atoi(argv[6]);
    dargs[10] = atoi(argv[2]);
    dargs[11] = atoi(argv[3]);
    dargs[12] = atoi(argv[4]);
  }
  if (argc > 8) {
    dargs[6] = atoi(argv[7]);
    dargs[12] = atoi(argv[8]);
  }
  printf("XXX N,H,W,C,R,S,Multipler,K: %d %d %d %d %d %d %d %d\n", dargs[0],
         dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6], dargs[12]);

  DepthwiseArgs args;
  args.batch = dargs[0];
  args.in_rows = dargs[1];
  args.in_cols = dargs[2];
  args.in_depth = dargs[3];
  args.filter_rows = dargs[4];
  args.filter_cols = dargs[5];
  args.depth_multiplier = dargs[6];
  args.stride = dargs[7];
  args.pad_rows = dargs[8];
  args.pad_cols = dargs[9];
  args.out_rows = dargs[10];
  args.out_cols = dargs[11];
  args.out_depth = dargs[12];

  int num_output = args.batch * args.out_depth * args.out_rows * args.out_cols;
  int num_input = args.batch * args.in_depth * args.in_rows * args.in_cols;
  int num_filter= args.in_depth * args.filter_rows * args.filter_cols *
                  args.depth_multiplier;

  int output_bytes = sizeof(float) * num_output;
  int input_bytes = sizeof(float) * num_input;
  int filter_bytes = sizeof(float) * num_filter;

  float *output;
  float *input;
  float *filter;

  checkCUDA(cudaMalloc(&output, output_bytes));
  checkCUDA(cudaMalloc(&input, input_bytes));
  checkCUDA(cudaMalloc(&filter, filter_bytes));

  init_array(filter, num_filter);
  init_array(input, num_input);
  // TODO really need this?
  init_array(output, num_output, 0.0);

  auto device_fn = DepthwiseConv2dGPUKernelNCHW<float, -1, -1, -1>;
  int blocks = 246;
  int threads = 512;
  checkCUDA(
      cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, device_fn, 0, 0));
  const int max_block_count = std::numeric_limits<int>::max();
  blocks = std::min(max_block_count, blocks);
  printf("XXX blocks: %d\n", blocks);
  printf("XXX threads: %d\n", threads);

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
  printf("XXX time(ms): %f\n", milliseconds / repeats);

  print_array(output, num_output, "Results:");
}

