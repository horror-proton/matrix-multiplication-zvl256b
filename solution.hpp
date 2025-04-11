#include <vector>

#include <riscv_vector.h>

inline void mm_unroll_block(const std::vector<float> &A,
                            const std::vector<float> &B, std::vector<float> &C,
                            size_t M, size_t N, size_t P) {

  // using m256 = __attribute__((riscv_rvv_vector_bits(256))) vfloat32m1_t;
  // using m512 = __attribute__((riscv_rvv_vector_bits(512))) vfloat32m2_t;
  // using m1024 = __attribute__((riscv_rvv_vector_bits(1024))) vfloat32m4_t;
  using m2048 = __attribute__((riscv_rvv_vector_bits(2048))) vfloat32m8_t; // 64

  static constexpr size_t blk_size = 64;
  static constexpr size_t blk_size_2 = 16;

#pragma omp parallel for
  for (size_t i = 0; i < M; i += blk_size_2) {
    for (size_t j = 0; j < P; j += blk_size) {
      m2048 c_tmp[blk_size_2];
#pragma GCC unroll blk_size_2
      for (size_t ii = 0; ii < blk_size_2; ++ii)
        c_tmp[ii] = __riscv_vfmv_v_f_f32m8(0, blk_size);

      for (size_t k = 0; k < N; k += 1) {
        const auto b_strip =
            __riscv_vle32_v_f32m8(B.data() + (k * P) + j, blk_size);

#pragma GCC unroll blk_size_2
        for (size_t ii = 0; ii < blk_size_2; ++ii) {
          const auto a_v = A[((i + ii) * N) + k];
          c_tmp[ii] = __riscv_vfmacc(c_tmp[ii], a_v, b_strip, blk_size);
        }
      }

      auto *p_dst = C.data() + (i * P) + j;

#pragma GCC unroll blk_size_2
      for (size_t ii = 0; ii < blk_size_2; ++ii)
        __riscv_vse32(p_dst + (ii * P), c_tmp[ii], blk_size);
    }
  };
}
