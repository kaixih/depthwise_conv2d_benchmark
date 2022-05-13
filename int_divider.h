#define EIGEN_DEVICE_FUNC __host__ __device__

struct FastDividerUint32 {
  inline EIGEN_DEVICE_FUNC FastDividerUint32(uint32_t d) : divisor(d) {
    // We assume that the divisor is at most INT32_MAX, which is enough for our
    // purpose when d is any positive int value.
    assert(divisor >= 1 && divisor <= INT32_MAX);

    // The fast int division can substitute fast multiplication for slow
    // division. For detailed information see:
    //   https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
    //
    // Basics: the int division can be transformed to:
    //   n / d = (m * n) / 2^(32 + s)
    // where 'n' is the numerator and 'd' is the divisor. For a given 'd', we
    // need to find a magic number 'm' and a shift 's'.
    // (1). The shift 's' is calculated by log2ceil(d).
#if defined(__CUDA_ARCH__)
    shift = 32 - __clz(divisor - 1);
#else
    for (shift = 0; shift < 32; shift++) {
      if ((1U << shift) >= divisor) break;
    }
#endif

    // (2). The magic number 'm' is defined as:
    //    m = 2^(32 + s) / d + 1
    // Note, the last '1' is to round up (which will be rounded down later by
    // dividing two). In practice, however, 'm' is a 33-bit value. To fit the
    // 32-bit range, we introduce:
    //   magic = m - 2^32, where magic is guaranteed to be 32bit.
    //         = 2^(32 + s) / d - 2^32 + 1
    //         = 2^32 * 2^s / d - 2^32 * d / d + 1
    //         = (2^32 * (2^s - d)) / d + 1
    uint64_t m = (0x100000000ull * ((0x1ull << shift) - divisor)) / divisor + 1;
    magic = static_cast<uint32_t>(m);
  }

  inline EIGEN_DEVICE_FUNC operator uint32_t() const {
		return divisor;
	}

  uint32_t divisor;
  uint32_t magic;
  uint32_t shift;
};

inline EIGEN_DEVICE_FUNC uint32_t operator/(const uint32_t n,
                                            const FastDividerUint32& fdiv) {
  // (3). We will use the 32-bit 'magic' in the formula:
  //   n / d = (m * n) / 2^(32 + s)
  //         = (magic + 2^32) * n / 2^(32 + s)
  //         = (magic * n) / 2^(32 + s) + n / 2^s
  //         = (magic * n) / 2^32 / 2^s + n / 2^s
  //         = (magic * n / 2^32 + n) / 2^s
#if defined(__CUDA_ARCH__)
  uint32_t q = __umulhi(n, fdiv.magic);
#else
  uint32_t q = static_cast<uint32_t>(
                   (static_cast<uint64_t>(n) * fdiv.magic) >> 32);
#endif
  return (n + q) >> fdiv.shift;
}

inline EIGEN_DEVICE_FUNC uint32_t operator%(const uint32_t n,
                                            const FastDividerUint32& fdiv) {
  return n - (n / fdiv) * fdiv.divisor;
}

inline EIGEN_DEVICE_FUNC uint32_t operator/(const int n,
                                            const FastDividerUint32& fdiv) {
  return static_cast<uint32_t>(n) / fdiv;
}

inline EIGEN_DEVICE_FUNC uint32_t operator%(const int n,
                                            const FastDividerUint32& fdiv) {
  return static_cast<uint32_t>(n) % fdiv;
}
