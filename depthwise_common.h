
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

void set_up(int argc, char **argv, DepthwiseArgs &args) {
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
  printf(">>> LOG: N,H,W,C,R,S,Multipler,K: %d %d %d %d %d %d %d %d\n",
         dargs[0], dargs[1], dargs[2], dargs[3], dargs[4], dargs[5], dargs[6],
         dargs[12]);

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
}

template <typename T>
__host__ __device__ inline const T& tf_min(const T& x, const T& y) {
  return x < y ? x : y;
}

template <typename T>
__host__ __device__ inline const T& tf_max(const T& x, const T& y) {
  return x < y ? y : x;
}

// Overloads of the above functions for float and double.
__host__ __device__ inline float tf_min(float x, float y) {
  return fminf(x, y);
}
__host__ __device__ inline double tf_min(double x, double y) {
  return fmin(x, y);
}
__host__ __device__ inline float tf_max(float x, float y) {
  return fmaxf(x, y);
}
__host__ __device__ inline double tf_max(double x, double y) {
  return fmax(x, y);
}
