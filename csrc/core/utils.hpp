#pragma once

#include <climits>
#include <cstdint>
#include <cstdlib>
#ifdef _MSC_VER
  #include <bit>
#endif

// [edge-infer W2] __builtin_clz is a GCC/Clang builtin; MSVC has no such
// identifier. std::countl_zero (C++20, <bit>) is the standard spelling and is
// constexpr, which this function needs. Same result on every compiler, so the
// GCC path is kept only for toolchains predating C++20 support here.
inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
#ifdef _MSC_VER
  return 1u << (CHAR_BIT * sizeof(num) - std::countl_zero(num - 1));
#else
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
#endif
}

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Round a down to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}

namespace vllm {

// Returns true if env VLLM_BATCH_INVARIANT=1
inline bool vllm_is_batch_invariant() {
  static bool cached = []() {
    const char* val = std::getenv("VLLM_BATCH_INVARIANT");
    return val && std::atoi(val) != 0;
  }();
  return cached;
}

}  // namespace vllm
