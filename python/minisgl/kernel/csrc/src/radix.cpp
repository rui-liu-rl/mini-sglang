// =====================================================================================
// FILE MAP -- kernel/csrc/src/radix.cpp
// =====================================================================================
// WHAT THIS FILE IS: one small C++ function, `fast_compare_key`, exported to Python
//                    (see kernel/radix.py).
//
// ELI5: "Where do these two lists of numbers first disagree?" Used by the radix prefix cache to
//       ask how many leading token ids a tree branch shares with a new order. std::mismatch walks
//       both lists together and stops at the first difference, much faster than a Python loop.
//
// EXAMPLE: a = [5, 6, 7, 8], b = [5, 6, 9]  ->  only the first 3 entries are compared; the first
//          difference is at index 2, so the answer is 2.
//          a = [5, 6], b = [5, 6, 7]  ->  2 (a is a prefix of b).
//
// RULES (checked first, with a readable error): both tensors 1-D, contiguous, on the CPU, int32 or
//       int64, and of the SAME dtype.
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

// WHAT: is this a flat, contiguous list of 32- or 64-bit integers living in CPU memory?
auto _is_1d_cpu_int_tensor(const tvm::ffi::TensorView tensor) -> bool {
  return tensor.ndim() == 1 && tensor.is_contiguous() &&
         tensor.device().device_type == kDLCPU &&
         (tensor.dtype().code == kDLInt) &&
         (tensor.dtype().bits == 32 || tensor.dtype().bits == 64);
}

// WHAT: index of the first position where a and b differ, looking only at the shorter length.
auto fast_compare_key(const tvm::ffi::TensorView a,
                      const tvm::ffi::TensorView b) -> size_t {
  host::RuntimeCheck(_is_1d_cpu_int_tensor(a) && _is_1d_cpu_int_tensor(b),
                     "Both tensors must be 1D CPU int tensors.");
  host::RuntimeCheck(a.dtype() == b.dtype());
  const auto a_ptr = a.data_ptr();
  const auto b_ptr = b.data_ptr();
  const auto common_len = std::min(a.size(0), b.size(0));
  // WHAT: the two branches differ only in the integer width used to read the memory.
  if (a.dtype().bits == 64) {
    const auto a_ptr_64 = static_cast<const int64_t *>(a_ptr);
    const auto b_ptr_64 = static_cast<const int64_t *>(b_ptr);
    const auto diff_pos =
        std::mismatch(a_ptr_64, a_ptr_64 + common_len, b_ptr_64);
    return static_cast<size_t>(diff_pos.first - a_ptr_64);
  } else {
    const auto a_ptr_32 = static_cast<const int32_t *>(a_ptr);
    const auto b_ptr_32 = static_cast<const int32_t *>(b_ptr);
    const auto diff_pos =
        std::mismatch(a_ptr_32, a_ptr_32 + common_len, b_ptr_32);
    return static_cast<size_t>(diff_pos.first - a_ptr_32);
  }
}

} // namespace

// WHAT: make the function callable from Python as module.fast_compare_key(a, b).
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_key, fast_compare_key);
