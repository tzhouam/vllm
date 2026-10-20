#pragma once
// [edge-infer W2] Force-included into every translation unit on Windows via
// /FI (MSVC) and --pre-include (nvcc).
//
// vLLM's CUDA sources assume a GNU/Linux toolchain in many small, independent
// ways. Patching them one file at a time costs a full rebuild per discovery,
// so the whole-class fixes live here instead. Everything below is either a
// standard spelling MSVC omits or a typedef POSIX provides and C++ does not;
// none of it changes behaviour on a compiler that already has these.

#ifdef _MSC_VER

  // MSVC hides M_PI, M_SQRT2 and friends behind this.
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif

  // `uint` is a POSIX typedef from <sys/types.h>, not standard C++.
  #include <cstddef>
typedef unsigned int uint;

  // ssize_t is POSIX; Windows spells the same thing SSIZE_T.
  #include <BaseTsd.h>
typedef SSIZE_T ssize_t;

  // [edge-infer W7] CUtensorMap alignment. CUDA 13's cuda.h declares the TMA
  // descriptor `alignas(128)` (CUDA 12 said 64). Every CUTLASS TMA kernel takes
  // its Params -- which embed CUtensorMap -- as a by-value __global__
  // parameter, and nvcc's host pass declares that same by-value signature for
  // cl.exe, which rejects any parameter aligned above 64 bytes (C2719,
  // measured: 16/32/64 pass, 128 fails). __grid_constant__ does not change the
  // host signature (measured). So under MSVC the only way to compile a TMA
  // kernel is for CUtensorMap to be 64-byte aligned, consistently in the host
  // and device passes. The documented requirement is 64: cuTensorMapEncode* say
  // "tensorMap address must be aligned to 64 bytes", and CUDA 12 shipped
  // exactly that. Done by including cuda.h once here with `alignas` rewritten
  // for the duration of that include only; it is the header's sole alignas.
  // Whether 64 is also enough for the hardware on sm_120 is what
  // verify_cutlass.py's FP8 GEMM check answers, not this comment.
  #if __has_include(<cuda.h>)
    #pragma push_macro("alignas")
    #define alignas(x) alignas(64)
    #include <cuda.h>
    #pragma pop_macro("alignas")
    #define EDGE_INFER_CUTENSORMAP_ALIGN64 1
  #endif

#endif  // _MSC_VER
