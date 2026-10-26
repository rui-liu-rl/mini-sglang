// =====================================================================================
// FILE MAP -- kernel/csrc/jit/index.cu
// =====================================================================================
// WHAT THIS FILE IS: the GPU program behind kernel/index.py: gather rows of a table by id.
//
//     output[i, :] = weight[ indices[i], :]            (the plain version, index_kernel)
//     output[i, :] = weight[ indices[i] - start, :]    if start <= indices[i] < start + length
//                    zeros                              otherwise        (masked_index_kernel)
//
// ELI5: the embedding lookup. For each word id in the list, copy that word's row from the big
//       table into the answer. The masked version is for tensor parallelism, where this GPU owns
//       only rows [start, start + length) of the vocabulary: ids outside that window yield a row
//       of zeros so a later all_reduce (sum) across GPUs fills in the true row.
//
// WORK DIVISION -- ONE WARP PER (ROW PIECE):
//       a row of `element_size` bytes is cut into `num_splits` pieces of kSizePerWarp bytes;
//       each piece is copied by one warp (32 threads, see warp.cuh). So for L ids there are
//       L * num_splits warps, packed kNumThreads/32 warps per block.
//
//          warp_id -> id number = warp_id / num_splits, piece number = warp_id % num_splits
//          src = weight + id * element_size + piece * kSizePerWarp
//          dst = output + warp_id * kSizePerWarp          (output rows are contiguous)
//
//       Example (Qwen3-0.6B, bf16): element_size = 1024 x 2 = 2048 bytes, num_splits = 4, so each
//       warp copies 512 bytes (one round of 16-byte handfuls). 10 ids -> 40 warps -> 10 blocks of
//       4 warps (128 threads).
//
// TEMPLATE PARAMETERS (all compile-time constants, baked in by kernel/index.py):
//       element_size, num_splits, num_threads, max_concurrency (blocks per SM), use_pdl.
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

// WHAT: the arguments passed to the plain kernel (one struct, passed by value as a "grid constant"
//       so every thread reads it from fast constant memory).
struct IndexKernelParams {
  void *__restrict__ output;
  const void *__restrict__ weight;
  const void *__restrict__ indice;
  std::size_t num_warps;
};

// WHAT: the plain arguments plus the window of rows this GPU owns.
struct MaskedKernelParams {
  IndexKernelParams params;
  std::size_t start;
  std::size_t length;
};

// ELI5: the plain lookup. Each warp: "which id am I serving? copy my piece of that row."
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    index_kernel(const __grid_constant__ IndexKernelParams params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  // WHAT: compile-time sanity checks on the settings.
  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  // WHAT: my global warp number = (warp within the block) + (block number) x (warps per block).
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  // WHAT: the last block may contain more warps than there is work; extra warps do nothing.
  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits];
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    const auto src = pointer::offset(weight, pos * kSize,
                                     (warp_id % kNumSplits) * kSizePerWarp);
    warp::copy<kSizePerWarp>(dst, src);
  }

  PDL::launch<kUsePDL>();
}

// ELI5: the tensor-parallel lookup: only rows in [start, start + length) are mine.
// GOTCHA: `pos = indices[...] - start` mixes a signed id with an unsigned std::size_t `start`, so a
//         negative result (id below my window) wraps around to a huge unsigned number; I read
//         `pos < length` as relying on that wrap-around to reject ids below `start`.
//         Ids at or beyond start + length are rejected by the same comparison.
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t kNumSplits, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    masked_index_kernel(
        const __grid_constant__ MaskedKernelParams mask_params) {
  using namespace device;
  constexpr auto kSize = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;
  constexpr auto kWarpPerBlock = static_cast<unsigned>(kNumThreads / 32);

  static_assert(kNumThreads % 32 == 0);
  static_assert(std::has_single_bit(kNumSplits));
  static_assert(kElementSize % kNumSplits == 0);

  const auto &[params, start, length] = mask_params;
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  PDL::wait<kUsePDL>();

  if (warp_id < num_warps) {
    const auto pos = indices[warp_id / kNumSplits] - start;
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);
    if (pos < length) {
      const auto src = pointer::offset(weight, pos * kSize,
                                       (warp_id % kNumSplits) * kSizePerWarp);
      warp::copy<kSizePerWarp>(dst, src);
    } else {
      warp::reset<kSizePerWarp>(dst);
    }
  }

  PDL::launch<kUsePDL>();
}

// ELI5: the host-side launcher (runs on the CPU). Python calls IndexKernel<...>::run(weights,
//       indices, output, mask). It (1) checks the tensors, (2) works out how many warps and
//       blocks are needed, (3) picks the kernel variant (plain or masked; int32 or int64 ids)
//       and launches it on the current CUDA stream.
template <std::size_t element_size,   // depends on data type and embedding dim
          std::size_t num_splits = 1, // how many warps handles one element
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct IndexKernel {
  static void run(const tvm::ffi::TensorView weights,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView output,
                  tvm::ffi::Optional<tvm::ffi::Tuple<int, int>> mask_opts) {
    using namespace host;
    auto D = SymbolicSize{"D"}; // embedding size
    auto L = SymbolicSize{"L"}; // num indices
    auto device_ = SymbolicDevice{};
    auto weights_dtype_ = SymbolicDType{};
    auto indices_dtype_ = SymbolicDType{};

    // WHAT: weights [any rows, D]; output [L, D] with the same dtype and device; indices [L] int32/int64.
    //       D and L are learned from the first tensor that mentions them and enforced on the others.
    TensorMatcher({-1, D}) //
        .with_dtype(weights_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(weights);
    TensorMatcher({L, D}) //
        .with_dtype(weights_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(output);
    TensorMatcher({L}) //
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(indices);

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto num_indices = L.unwrap();
    const auto entry_size = dtype_bytes(weights_dtype_.unwrap()) * D.unwrap();
    // WHAT: this compiled variant was built for ONE row width; refuse a different one.
    RuntimeCheck(entry_size == element_size,
                 "IndexKernel: element_size mismatch.");

    constexpr auto kWarpPerBlock = num_threads / 32;
    const auto num_warps = num_splits * num_indices;
    const auto num_blocks = div_ceil(num_warps, kWarpPerBlock);
    const auto params = IndexKernelParams{
        .output = static_cast<char *>(output.data_ptr()),
        .weight = static_cast<const char *>(weights.data_ptr()),
        .indice = indices.data_ptr(),
        .num_warps = num_warps,
    };

    if (mask_opts.has_value()) {
      // WHAT: Python passed (start, length): use the masked kernel.
      const auto &obj = mask_opts.value();
      const auto [start, length] = obj;
      const auto m_params = MaskedKernelParams{
          .params = params,
          .start = static_cast<std::size_t>(start),
          .length = static_cast<std::size_t>(length),
      };
      const auto kernel =
          use_int32 ? masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, int32_t>
                    : masked_index_kernel<num_threads, max_concurrency, use_pdl,
                                          element_size, num_splits, int64_t>;
      LaunchKernel(num_blocks, num_threads, device)
          .with_attr(use_pdl)(kernel, m_params);
    } else {
      const auto kernel =
          use_int32 ? index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, int32_t>
                    : index_kernel<num_threads, max_concurrency, use_pdl,
                                   element_size, num_splits, int64_t>;
      LaunchKernel(num_blocks, num_threads, device)
          .with_attr(use_pdl)(kernel, params);
    }
  }
};

} // namespace
