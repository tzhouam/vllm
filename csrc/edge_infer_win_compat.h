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

#endif  // _MSC_VER
