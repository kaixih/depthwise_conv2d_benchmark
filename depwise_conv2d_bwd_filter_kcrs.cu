#include <iostream>

#define checkCUDA(expression)                                                  \
  {                                                                            \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudaGetErrorString(status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
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
  int pad_rows; // Amount of padding to the top of the input
  int pad_cols; // Amount of padding to the left of the input

  // Output layer dimensions
  int out_rows;
  int out_cols;
  int out_depth;

  // XXX
  int test_factor;

  DepthwiseArgs()
      : batch(0), in_rows(0), in_cols(0), in_depth(0), filter_rows(0),
        filter_cols(0), depth_multiplier(0), stride(0), pad_rows(0),
        pad_cols(0), out_rows(0), out_cols(0), out_depth(0) {}
};

template <typename T> class GpuGridRange {
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

template <typename T> __device__ GpuGridRange<T> GpuGridRangeX(T count) {
  return GpuGridRange<T>(
      /*begin=*/blockIdx.x * blockDim.x + threadIdx.x,
      /*delta=*/gridDim.x * blockDim.x, /*end=*/count);
}

template <typename T> __host__ __device__ T GpuLdg(const T *address) {
  return __ldg(address);
}

template <typename T> __host__ __device__ inline T ldg(const T *ptr) {
  return GpuLdg(ptr);
}

#define GPU_1D_KERNEL_LOOP(i, n) for (int i : GpuGridRangeX<int>(n))
#define UNROLL _Pragma("unroll")
#define NOUNROLL _Pragma("nounroll")

template <typename T> struct CudaSupportedTypeImpl { using type = T; };

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

template <typename DType, int kFilterWidth, int kFilterHeight>
__global__ void __launch_bounds__(640, 2)
    DepthwiseConv2dBackwardFilterKernel(const DepthwiseArgs args,
                                        const DType *out_backprop,
                                        const DType *input,
                                        DType *filter_backprop,
                                        int num_out_backprop) {
  const int in_channel = args.in_depth;
  const int in_height = args.in_rows;
  const int in_width = args.in_cols;
  const int filter_height =
      kFilterHeight > 0 ? kFilterHeight : args.filter_rows;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_cols;
  const int stride_height = args.stride;
  const int stride_width = args.stride;
  const int pad_height = args.pad_rows;
  const int pad_width = args.pad_cols;
  const int out_channel = args.out_depth;
  const int out_height = args.out_rows;
  const int out_width = args.out_cols;

  GPU_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % out_channel;
    const int out_b = thread_id / out_width / out_height / out_channel;
    const int in_c = out_c;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_row_start = out_h * stride_height - pad_height;
    const int in_col_start = out_w * stride_width - pad_width;
    const int in_row_end = in_row_start + filter_height;
    const int in_col_end = in_col_start + filter_width;

    const int out_backprop_offset =
        (out_b * out_channel * out_height * out_width) +
        (out_c * out_height * out_width) + (out_h * out_width) + (out_w);

    const DType out_bp = ldg(out_backprop + out_backprop_offset);
    if (in_row_start >= 0 && in_col_start >= 0 && in_row_end < in_height &&
        in_col_end < in_width) {
      UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_row = in_row_start + f_h;
        // Avoid repeated computation.
        const int input_offset_temp =
            (out_b * in_channel * in_height * in_width) +
            (in_c * in_height * in_width) + (in_row * in_width);
        const int filter_backprop_temp =
            (in_c * filter_width * filter_height) + (filter_width * f_h);

        UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_col = in_col_start + f_w;
          const int input_offset = input_offset_temp + in_col;
          DType partial_sum = ldg(input + input_offset) * out_bp;
          DType *addr = filter_backprop + (filter_backprop_temp + f_w);
#ifdef DEBUG_NON_ATOMIC
          *addr = partial_sum;
#else
          GpuAtomicAdd(addr, partial_sum);
#endif
        }
      }
    } else {
      UNROLL for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_row = in_row_start + f_h;
        // Avoid repeated computation.
        const int input_offset_temp =
            (out_b * in_channel * in_height * in_width) +
            (in_c * in_height * in_width) + (in_row * in_width);
        const int filter_backprop_temp =
            (in_c * filter_width * filter_height) + (filter_width * f_h);
        UNROLL for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_col = in_col_start + f_w;

          if (in_row >= 0 && in_row < in_height && in_col >= 0 &&
              in_col < in_width) {
            const int input_offset = input_offset_temp + in_col;
            DType partial_sum = ldg(input + input_offset) * out_bp;
            DType *addr = filter_backprop + (filter_backprop_temp + f_w);
            // Potentially many threads can add to the same address so we have
            // to use atomic add here.
            // TODO(jmchen): If atomic add turns out to be slow, we can:
            // 1. allocate multiple buffers for the gradients (one for each
            // example in a batch, for example). This can reduce the
            // contention on the destination; 2. Have each thread compute one
            // gradient for an element in the filters. This should work well
            // when the input depth is big and filter size is not too small.
#ifdef DEBUG_NON_ATOMIC
            *addr = partial_sum;
#else
            GpuAtomicAdd(addr, partial_sum);
#endif
          }
        }
      }
    }
  }
}

template <typename T> void init_array(T *dev_ptr, int n) {
  T *host_ptr = new T[n];
  for (int i = 0; i < n; i++) {
    host_ptr[i] = 0.1f;
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
    if ((i + 1) % 10 == 0)
      break;
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
  printf("XXX N,H,W,C,R,S: %d %d %d %d %d %d\n", dargs[0], dargs[1], dargs[2],
         dargs[3], dargs[4], dargs[5]);

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

  int num_out_backprop =
      args.batch * args.out_depth * args.out_rows * args.out_cols;
  int num_input = args.batch * args.in_depth * args.in_rows * args.in_cols;
  int num_filter_backprop = args.in_depth * args.filter_rows * args.filter_cols;

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

  int blocks = 246;
  int threads = 512;

  auto device_fn = DepthwiseConv2dBackwardFilterKernel<float, -1, -1>;
  checkCUDA(
      cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, device_fn, 0, 640));
  blocks = std::min(blocks, DivUp(num_out_backprop, threads));
  printf("XXX blocks,threads: %d %d\n", blocks, threads);

  auto launcher = [&](int repeats) {
    for (int i = 0; i < repeats; i++) {
      device_fn<<<blocks, threads>>>(args, out_backprop, input, filter_backprop,
                                     num_out_backprop);
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

  print_array(filter_backprop, num_filter_backprop, "Results:");
}
