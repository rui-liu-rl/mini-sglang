// =====================================================================================
// FILE MAP -- kernel/csrc/include/minisgl/utils.cuh
// =====================================================================================
// WHAT THIS FILE IS: the shared toolbox for GPU code (the ".cuh" = CUDA header), in two halves:
//
//   namespace device (code that runs ON the GPU):
//       kWarpThreads        32: the number of threads in a warp
//       div_ceil            integer division rounding up
//       pointer::offset     byte-level pointer arithmetic, allowed ONLY on void* (a safety rule)
//       PDL::wait/launch    hooks for "programmatic dependent launch" (see below)
//
//   namespace host (code that runs on the CPU and talks to the GPU):
//       CUDA_CHECK          turn a CUDA error code into a thrown error with file and line
//       set_smem_once       ask for more shared memory for a kernel, once
//       LaunchKernel        the one way every kernel here is launched
//
// ELI5: the kernels in jit/ and src/ all need the same boring helpers: launch a kernel on the
//       right GPU lane, complain clearly when CUDA fails, move a pointer by some bytes. They live
//       here so each kernel file can focus on its one job.
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#pragma once

#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <source_location>
#include <type_traits>

namespace device {

// WHAT: GPU threads are executed in groups of 32 called warps.
inline constexpr auto kWarpThreads = 32u;

// WHAT: ceiling division that can be called from GPU code. div_ceil(13, 4) = 4.
template <std::integral T, std::integral U>
__always_inline __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

// WHY: adding to a typed pointer silently scales by the element size, a classic source of bugs.
//      Forcing void* means every offset below is in BYTES, written out explicitly:
//      offset(base, row * row_bytes, split * piece_bytes) = base + row*row_bytes + split*piece_bytes.
template <typename T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

// WHAT: the same for read-only pointers.
template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

// ELI5: Programmatic Dependent Launch (a Hopper-and-newer GPU feature): a kernel can tell the
//       NEXT kernel in the same lane "you may start getting ready now" instead of waiting until
//       this one completely ends. wait() = "before I read what the previous kernel wrote, wait
//       until it says it is done"; launch() = "the next kernel may begin its warm-up".
//       When the template flag is false (as for the kernels here, KernelConfig.use_pdl=False),
//       both compile to nothing.
namespace PDL {

template <bool kUsePDL> __always_inline __device__ void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL> __always_inline __device__ void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}

} // namespace PDL

} // namespace device

namespace host {

// ELI5: "Did that CUDA call work? If not, say exactly where and why."
// WHAT: if `error` is not cudaSuccess, throw an error that names the file and line (std::source_location
//       fills those in automatically) and CUDA's own message.
inline auto
CUDA_CHECK(::cudaError_t error,
           std::source_location location = std::source_location::current())
    -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

// WHAT: the same check, but asks CUDA "what was the last error?" (used right after a kernel launch).
inline auto
CUDA_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return CUDA_CHECK(::cudaGetLastError(), location);
}

// WHAT: a kernel that needs more than the default 48 KiB of dynamic shared memory must opt in once.
//       This does the opt-in the first time it is called (the `static const` runs once) and then
//       refuses any later request for MORE than the first, because the attribute was only set to
//       the first size.
template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    CUDA_CHECK(::cudaFuncSetAttribute(
        F, ::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  RuntimeCheck(
      smem_size <= last_smem_size,
      "Dynamic shared memory size exceeds the previously set maximum size: ",
      last_smem_size, " bytes");
}

// ELI5: the launch pad. Usage:   LaunchKernel(num_blocks, num_threads, device)(kernel, params);
//       It builds a launch configuration (grid size, block size, dynamic shared memory, which
//       GPU lane) and then starts the kernel with cudaLaunchKernelEx, checking for errors.
// WHY the lane comes from TVM-FFI: resolve_device() asks the FFI layer for "the current CUDA stream
//     of this device", which is the stream PyTorch is currently using. So the kernel runs in the
//     same lane as the surrounding PyTorch operations and is correctly ordered with them.
// with_attr(use_pdl): switches on the programmatic-stream-serialization launch attribute when PDL is on.
struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device),
                               dynamic_shared_mem_bytes)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream,
                               dynamic_shared_mem_bytes)) {}

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    CUDA_CHECK(
        ::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
  }

  auto with_attr(bool use_pdl) -> LaunchKernel & {
    if (use_pdl) {
      m_attr_cache.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr_cache;
      m_config.numAttrs = 1;
    } else {
      m_config.numAttrs = 0;
    }
    return *this;
  }

private:
  static auto s_make_config(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                            std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }
  cudaLaunchConfig_t m_config;
  cudaLaunchAttribute m_attr_cache;
};

} // namespace host
