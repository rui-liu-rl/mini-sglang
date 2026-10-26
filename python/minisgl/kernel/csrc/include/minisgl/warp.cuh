// =====================================================================================
// FILE MAP -- kernel/csrc/include/minisgl/warp.cuh
// =====================================================================================
// WHAT THIS FILE IS: two tiny GPU helpers that make ONE WARP (32 GPU threads working in lockstep)
//                    copy or zero a block of bytes quickly: `warp::copy` and `warp::reset`.
//
// ELI5: Imagine 32 helpers standing in a row. To move 2,048 bytes, each helper grabs a 16-byte
//       handful at a time, and in each round the 32 helpers pick up 32 NEIGHBOURING handfuls (512
//       bytes in a row), so the memory system serves them in one big efficient gulp ("coalesced
//       access"). Four rounds move all 2,048 bytes.
//
//          bytes:   |h0 h1 h2 ... h31|h0 h1 h2 ... h31|h0 ...   (round 0, round 1, ...)
//                    helper i takes handful (round * 32 + i)
//
// HOW THE HANDFUL SIZE IS CHOSEN (resolve_unit_size): the widest of 16, 8 or 4 bytes such that
//       the total byte count is a multiple of (handful x 32). Example: 2048 bytes -> 16 (16 x 32 =
//       512, and 2048 / 512 = 4 rounds). 512 bytes -> 16 (one round). 384 bytes -> 4
//       (4 x 32 = 128; 384 / 128 = 3 rounds). A size that fits none triggers a compile-time error.
//       The handful type is uint4 (16 B), uint2 (8 B) or uint1 (4 B): one machine load/store.
//
// WHERE USED: kernel/csrc/jit/index.cu (copy a table row, or zero it) and store.cu (copy K and V rows).
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#pragma once
#include <minisgl/utils.cuh>

#include <sys/cdefs.h>

#include <cstddef>

namespace device::warp {

namespace details {

// WHAT: map a handful size in bytes to the matching 4-, 8- or 16-byte type.
template <std::size_t kUnit> inline constexpr auto get_mem_package() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4,
                  "Unsupported memory package size");
  }
}

// WHAT: pick the widest handful that divides the work evenly across the 32 threads (see the
//       header comment). Returns 0 if none fits, which makes get_mem_package fail to compile.
inline constexpr auto resolve_unit_size(std::size_t x) -> std::size_t {
  if (x % (16 * kWarpThreads) == 0)
    return 16;
  if (x % (8 * kWarpThreads) == 0)
    return 8;
  if (x % (4 * kWarpThreads) == 0)
    return 4;
  return 0; // trigger static assert in _get_mem_package
}

template <std::size_t kBytes, std::size_t kUnit>
using mem_package_t = decltype(get_mem_package<kUnit>());

} // namespace details

// ELI5: the whole warp copies kBytes from src to dst, handful by handful.
// WHAT: bytes per round = handful size x 32 threads; rounds = kBytes / that. The loop is fully
//       unrolled (kLoopCount is a compile-time constant). Every thread must call this together
//       (it uses threadIdx.x % 32 as the lane number), and dst/src must not overlap (__restrict__).
template <std::size_t kBytes,
          std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void copy(void *__restrict__ dst,
                                     const void *__restrict__ src) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0,
                "kBytes must be multiple of 128 bytes");

  const auto dst_packed = static_cast<Package *>(dst);
  const auto src_packed = static_cast<const Package *>(src);
  const auto lane_id = threadIdx.x % kWarpThreads;

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    const auto j = i * kWarpThreads + lane_id;
    dst_packed[j] = src_packed[j];
  }
}

// ELI5: the same walk, but writing zeros instead of copying. Used to blank a row whose id belongs
//       to another GPU's share of the vocabulary (index.cu, masked_index_kernel).
template <std::size_t kBytes,
          std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void reset(void *__restrict__ dst) {
  using Package = details::mem_package_t<kBytes, kUnit>;
  constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;
  static_assert(kBytes % kBytesPerLoop == 0,
                "warp_copy: kBytes must be multiple of 128 bytes");

  const auto dst_ = static_cast<Package *>(dst);
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto zero_value = Package{};

#pragma unroll kLoopCount
  for (std::size_t i = 0; i < kLoopCount; ++i) {
    dst_[i * kWarpThreads + lane_id] = zero_value;
  }
}

} // namespace device::warp
