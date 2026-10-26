// =====================================================================================
// FILE MAP -- kernel/csrc/src/tensor.cpp
// =====================================================================================
// WHAT THIS FILE IS: a test target for the tensor-checking toolkit in include/minisgl/tensor.h.
//                    Python reaches it through kernel/tensor.py (test_tensor); tests/kernel/
//                    test_tensor.py is the only user.
//
// ELI5: Every GPU kernel in this package starts by checking its tensors ("right shape? right
//       type? on the GPU? contiguous?") so mistakes become clear error messages instead of
//       silent garbage. `TensorMatcher` is the checking tool. This function exercises it:
//
//         x must be:  shape [N, 1024]; the last dimension contiguous (stride 1), the row stride
//                     anything (-1 means "any"); dtype int32 or float32; on the CPU
//         y must be:  shape [N, 1024] with the SAME N (N is a named unknown, "SymbolicSize");
//                     contiguous (the default); dtype int32 or int64; on a CUDA device
//                     (written as {kDLCUDA, 1}, which I read as device id 1)
//         and finally N must be a multiple of 4.
//
//       Any violation throws an error that says which condition failed.
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#include <minisgl/tensor.h>
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>

namespace {

auto test(tvm::ffi::TensorView x, tvm::ffi::TensorView y) -> void {
  // WHAT: N is "some number I will learn from the first tensor and then insist on for the second".
  auto N = host::SymbolicSize{"N"};
  const auto M = 1024;
  host::TensorMatcher({N, M})
      .with_strides({-1, 1}) // -1 means any
      .with_dtype<int, float>()
      .with_device<kDLCPU>()
      .verify(x);
  host::TensorMatcher({N, M}) // default contiguous
      .with_dtype({{kDLInt, 32, 1}, {kDLInt, 64, 1}})
      .with_device({{kDLCUDA, 1}})
      .verify(y);
  host::RuntimeCheck(N.unwrap() % 4 == 0);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(test, test);
