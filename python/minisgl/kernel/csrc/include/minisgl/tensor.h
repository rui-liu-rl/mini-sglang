// =====================================================================================
// FILE MAP -- kernel/csrc/include/minisgl/tensor.h
// =====================================================================================
// WHAT THIS FILE IS: `TensorMatcher`, a small declarative checker for the tensors a kernel
//                    receives from Python: "shape [L, D], strides [Y, 1], bf16 or fp16, on a CUDA
//                    device" in one readable chain. It also produces good error messages.
//
// ELI5: Every kernel wrapper begins with a bouncer at the door. Instead of a dozen if-statements,
//       the bouncer reads like a form:
//
//           auto D = host::SymbolicSize{"D"};   // "some number, to be learned"
//           auto L = host::SymbolicSize{"L"};
//           host::TensorMatcher({L, D})         // two-dimensional
//               .with_strides({-1, 1})          // row stride: anything; last dimension: contiguous
//               .with_dtype<int32_t, int64_t>() // one of these dtypes
//               .with_device<kDLCUDA>()         // on a CUDA GPU
//               .verify(tensor);                // do the checking now
//
//       The trick is the SYMBOLIC values: the FIRST tensor that mentions D teaches the checker
//       "D is 2048"; every later tensor that mentions D must then have 2048, or the check fails
//       with "Size mismatch for D(shape#1): expected 2048 but got 1024". Dtypes and devices work
//       the same way. After verification, `D.unwrap()` hands the learned number to the kernel.
//
// THE PIECES
//       SymbolicSize     a size that is unknown until the first match, then fixed
//       SymbolicDType    a dtype (or a set of allowed dtypes), fixed on first match
//       SymbolicDevice   a device (or allowed device types), fixed on first match
//       SizeRef/DTypeRef/DeviceRef   wrappers so an argument can be a literal (1024), a list of
//                        options, or a reference to one of the symbolic objects above
//       TensorMatcher    the chain: shape, optional strides, dtype, device, then verify
//
// GOTCHA: a SymbolicSize uses 0 to mean "not set yet", so a real size of 0 can never be bound
//         (unwrap() would then say "Size value is not set"). The kernels here never pass empty
//         tensors, so it does not matter in practice.
//
// WHO USES IT: jit/index.cu, jit/store.cu, src/tensor.cpp (the self-test).
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#pragma once
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <source_location>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace host {

namespace stdr = std::ranges;
namespace stdv = std::views;

namespace details {

struct SizeRef;
struct DTypeRef;
struct DeviceRef;

// WHAT: "wildcard" and "not set" markers used by the symbolic types below.
//       kAnyDeviceID: accept any device index; kAnySize (-1): accept any size;
//       kNullSize (0), kNullDType (code 18, not a real dtype), kNullDevice: "no value yet".
inline constexpr auto kAnyDeviceID = -1;
inline constexpr auto kAnySize = static_cast<int64_t>(-1);
inline constexpr auto kNullSize = static_cast<int64_t>(0);
inline constexpr auto kNullDType = static_cast<DLDataTypeCode>(18u);
inline constexpr auto kNullDevice = static_cast<DLDeviceType>(-1);

// WHAT: compile-time translation from a C++ type (int, float, int64_t, ...) to the DLPack dtype
//       description (code, bits, lanes) that tensors carry. `with_dtype<int, float>()` uses it.
template <typename T> struct dtype_trait {};

template <std::integral T> struct dtype_trait<T> {
  inline static constexpr auto value =
      DLDataType{.code = std::is_signed_v<T> ? DLDataTypeCode::kDLInt
                                             : DLDataTypeCode::kDLUInt,
                 .bits = static_cast<std::uint8_t>(sizeof(T) * 8),
                 .lanes = 1};
};

template <std::floating_point T> struct dtype_trait<T> {
  inline static constexpr auto value =
      DLDataType{.code = DLDataTypeCode::kDLFloat,
                 .bits = static_cast<std::uint8_t>(sizeof(T) * 8),
                 .lanes = 1};
};

template <DLDeviceType Code> struct device_trait {
  inline static constexpr auto value =
      DLDevice{.device_type = Code, .device_id = kAnyDeviceID};
};

// WHAT: the lists of allowed dtypes / device types, built at compile time from the template
//       arguments of with_dtype<...>() and with_device<...>().
template <typename... Ts>
inline constexpr auto kDTypeList =
    std::array<DLDataType, sizeof...(Ts)>{dtype_trait<Ts>::value...};

template <DLDeviceType... Codes>
inline constexpr auto kDeviceList =
    std::array<DLDevice, sizeof...(Codes)>{device_trait<Codes>::value...};

template <typename T> struct PrintAbleSpan {
  explicit PrintAbleSpan(std::span<const T> data) : data(data) {}
  std::span<const T> data;
};

// WHAT: printing helpers so error messages say "cuda[1]" or "[int32, int64]" instead of raw
//       numbers. kDeviceStringMap is a lookup table from DLPack's device-type code to a name.
// define DLDataType comparison and printing in root namespace
inline constexpr auto kDeviceStringMap = [] {
  constexpr auto map =
      std::array<std::pair<DLDeviceType, std::string_view>, 16>{
          std::pair{DLDeviceType::kDLCPU, "cpu"},
          std::pair{DLDeviceType::kDLCUDA, "cuda"},
          std::pair{DLDeviceType::kDLCUDAHost, "cuda_host"},
          std::pair{DLDeviceType::kDLOpenCL, "opencl"},
          std::pair{DLDeviceType::kDLVulkan, "vulkan"},
          std::pair{DLDeviceType::kDLMetal, "metal"},
          std::pair{DLDeviceType::kDLVPI, "vpi"},
          std::pair{DLDeviceType::kDLROCM, "rocm"},
          std::pair{DLDeviceType::kDLROCMHost, "rocm_host"},
          std::pair{DLDeviceType::kDLExtDev, "ext_dev"},
          std::pair{DLDeviceType::kDLCUDAManaged, "cuda_managed"},
          std::pair{DLDeviceType::kDLOneAPI, "oneapi"},
          std::pair{DLDeviceType::kDLWebGPU, "webgpu"},
          std::pair{DLDeviceType::kDLHexagon, "hexagon"},
          std::pair{DLDeviceType::kDLMAIA, "maia"},
          std::pair{DLDeviceType::kDLTrn, "trn"},
      };
  constexpr auto max_type = stdr::max(map | stdv::keys);
  auto result = std::array<std::string_view, max_type + 1>{};
  for (const auto &[code, name] : map) {
    result[static_cast<std::size_t>(code)] = name;
  }
  return result;
}();

struct PrintableDevice {
  DLDevice device;
};

inline auto &operator<<(std::ostream &os, DLDevice device) {
  const auto &mapping = kDeviceStringMap;
  const auto entry = static_cast<std::size_t>(device.device_type);
  RuntimeCheck(entry < mapping.size());
  const auto name = mapping[entry];
  RuntimeCheck(!name.empty(), "Unknown device: ", int(device.device_type));
  os << name;
  if (device.device_id != kAnyDeviceID)
    os << "[" << device.device_id << "]";
  return os;
}

inline auto &operator<<(std::ostream &os, PrintableDevice pd) {
  return os << pd.device;
}

template <typename T>
inline auto &operator<<(std::ostream &os, PrintAbleSpan<T> span) {
  os << "[";
  for (const auto i : stdv::iota(std::size_t{0}, span.data.size())) {
    if (i > 0) {
      os << ", ";
    }
    os << span.data[i];
  }
  os << "]";
  return os;
}

} // namespace details

// ELI5: a number that is unknown until the first tensor tells us, then must stay the same.
//       verify(size, ...): if already set, the new size must equal it, else it becomes the value.
// WHAT: non-copyable on purpose (copies would silently stop sharing the learned value).
struct SymbolicSize {
public:
  SymbolicSize(std::string_view annotation = {})
      : m_value(details::kNullSize), m_annotation(annotation) {}

  auto get_name() const -> std::string_view { return m_annotation; }
  auto set_value(int64_t value) -> void {
    RuntimeCheck(!this->has_value(), "Size value already set");
    m_value = value;
  }
  auto has_value() const -> bool { return m_value != 0; }
  auto get_value() const -> std::optional<int64_t> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }
  auto unwrap() const -> int64_t {
    RuntimeCheck(this->has_value(), "Size value is not set");
    return m_value;
  }

  SymbolicSize(const SymbolicSize &) = delete;
  SymbolicSize &operator=(const SymbolicSize &) = delete;

  auto verify(int64_t size, const char *prefix, int64_t dim) -> void {
    if (this->has_value()) {
      if (m_value != size) {
        [[unlikely]];
        Panic("Size mismatch for ", m_name(prefix, dim), ": expected ", m_value,
              " but got ", size);
      }
    } else {
      this->set_value(size);
    }
  }

  auto value_or_name(const char *prefix, int64_t dim) const -> std::string {
    if (const auto value = this->get_value()) {
      return std::to_string(*value);
    } else {
      return m_name(prefix, dim);
    }
  }

private:
  auto m_name(const char *prefix, int64_t dim) const -> std::string {
    const auto annotation = this->get_name();
    std::ostringstream os;
    if (annotation.empty()) {
      os << prefix << '#' << dim;
    } else {
      os << annotation << '(' << prefix << '#' << dim << ')';
    }
    return std::move(os).str();
  }

  std::int64_t m_value;
  std::string_view m_annotation;
};

inline auto operator==(DLDevice lhs, DLDevice rhs) -> bool {
  return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

// ELI5: the same idea for data types, plus an optional list of allowed ones
//       (set_options). The first tensor must be one of the allowed; later tensors must match it.
struct SymbolicDType {
public:
  SymbolicDType() : m_value({details::kNullDType, 0, 0}) {}

  auto set_value(DLDataType value) -> void {
    RuntimeCheck(!this->has_value(), "Dtype value already set");
    RuntimeCheck(
        m_check(value), "Dtype value [", value,
        "] not in the allowed options: ", details::PrintAbleSpan{m_options});
    m_value = value;
  }
  auto has_value() const -> bool { return m_value.code != details::kNullDType; }
  auto get_value() const -> std::optional<DLDataType> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }
  auto unwrap() const -> DLDataType {
    RuntimeCheck(this->has_value(), "Dtype value is not set");
    return m_value;
  }

  auto set_options(std::span<const DLDataType> options) -> void {
    m_options = options;
  }
  template <typename... Ts> auto set_options() -> void {
    m_options = details::kDTypeList<Ts...>;
  }

  auto verify(DLDataType dtype) -> void {
    if (this->has_value()) {
      RuntimeCheck(m_value == dtype, "DType mismatch: expected ", m_value,
                   " but got ", dtype);
    } else {
      this->set_value(dtype);
    }
  }

private:
  auto m_check(DLDataType value) const -> bool {
    return stdr::empty(m_options) ||
           (stdr::find(m_options, value) != stdr::end(m_options));
  }

  std::span<const DLDataType> m_options;
  DLDataType m_value;
};

// ELI5: the same idea for devices. An allowed option may leave the device index as a wildcard
//       (kAnyDeviceID): "any CUDA GPU"; once the first tensor is seen, later tensors must be on
//       exactly the same device.
struct SymbolicDevice {
public:
  SymbolicDevice() : m_value({details::kNullDevice, details::kAnyDeviceID}) {}

  auto set_value(DLDevice value) -> void {
    RuntimeCheck(!this->has_value(), "Device value already set");
    RuntimeCheck(
        m_check(value), "Device value [", details::PrintableDevice{value},
        "] not in the allowed options: ", details::PrintAbleSpan{m_options});
    m_value = value;
  }
  auto has_value() const -> bool {
    return m_value.device_type != details::kNullDevice;
  }
  auto get_value() const -> std::optional<DLDevice> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }
  auto unwrap() const -> DLDevice {
    RuntimeCheck(this->has_value(), "Device value is not set");
    return m_value;
  }

  auto set_options(std::span<const DLDevice> options) -> void {
    m_options = options;
  }
  template <DLDeviceType... Codes> auto set_options() -> void {
    m_options = details::kDeviceList<Codes...>;
  }

  auto verify(DLDevice device) -> void {
    if (this->has_value()) {
      RuntimeCheck(m_value == device, "Device mismatch: expected ",
                   details::PrintableDevice{m_value}, " but got ",
                   details::PrintableDevice{device});
    } else {
      this->set_value(device);
    }
  }

private:
  auto m_check(DLDevice value) const -> bool {
    return stdr::empty(m_options) ||
           (stdr::any_of(m_options, [value](const DLDevice &opt) {
             // device type must exactly match
             if (opt.device_type != value.device_type)
               return false;
             // device id can be wildcarded
             return opt.device_id == details::kAnyDeviceID ||
                    opt.device_id == value.device_id;
           }));
  }

  std::span<const DLDevice> m_options;
  DLDevice m_value;
};

struct TensorMatcher;

namespace details {

// ELI5: a pointer-like wrapper that either points at a symbolic object the caller owns (so several
//       tensors share it) or at a private one it creates for itself. SizeRef/DTypeRef/DeviceRef
//       below add the convenient constructors, so `{L, 1024}` or `{kDLInt, 32, 1}` just work.
template <typename T> struct BaseRef {
public:
  BaseRef(const BaseRef &) = delete;
  BaseRef &operator=(const BaseRef &) = delete;

  auto operator->() const -> T * { return m_ref; }
  auto operator*() const -> T & { return *m_ref; }
  auto rebind(T &other) -> void { m_ref = &other; }

  explicit BaseRef() : m_ref(&m_cache), m_cache() {}
  BaseRef(T &size) : m_ref(&size), m_cache() {}

private:
  T *m_ref;
  T m_cache;
};

struct SizeRef : BaseRef<SymbolicSize> {
  using BaseRef::BaseRef;
  SizeRef(int64_t value) {
    if (value != kAnySize) {
      (**this).set_value(value);
    } else {
      // otherwise, we can match any size
    }
  }
};

struct DTypeRef : BaseRef<SymbolicDType> {
  using BaseRef::BaseRef;
  DTypeRef(DLDataType options) { (**this).set_value(options); }
  DTypeRef(std::initializer_list<DLDataType> options) {
    (**this).set_options(options);
  }
  DTypeRef(std::span<const DLDataType> options) {
    (**this).set_options(options);
  }
};

struct DeviceRef : BaseRef<SymbolicDevice> {
  using BaseRef::BaseRef;
  DeviceRef(DLDevice options) { (**this).set_value(options); }
  DeviceRef(std::initializer_list<DLDevice> options) {
    (**this).set_options(options);
  }
  DeviceRef(std::span<const DLDevice> options) {
    (**this).set_options(options);
  }
};

} // namespace details

// ELI5: the checklist. Build it by chaining (shape in the constructor; then optionally
//       .with_strides, .with_dtype, .with_device), then call .verify(tensor) one or more times.
//       The chaining methods are `&&`-qualified: they can only be called on a temporary, which
//       enforces "write the whole chain in one expression".
// ABOUT STRIDES: if no strides are given, the tensor must be contiguous. A stride of -1 means
//       "any". A dimension of size 1 skips its stride check (its stride is meaningless).
// WHAT verify DOES ON FAILURE: it catches the low-level error and rethrows it with the full
//       description of what was expected (debug_str) and the file and line of the verify call.
struct TensorMatcher {
private:
  using SizeRef = details::SizeRef;
  using DTypeRef = details::DTypeRef;
  using DeviceRef = details::DeviceRef;
  using Loc_t = std::source_location;

public:
  TensorMatcher(const TensorMatcher &) = delete;
  TensorMatcher &operator=(const TensorMatcher &) = delete;

  explicit TensorMatcher(std::initializer_list<SizeRef> shape)
      : m_shape(shape), m_strides(), m_dtype() {}

  auto with_strides(std::initializer_list<SizeRef> strides) && //
      -> TensorMatcher && {
    // no partial update allowed
    RuntimeCheck(m_strides.size() == 0, "Strides already specified");
    RuntimeCheck(m_shape.size() == strides.size(),
                 "Strides size must match shape size");
    m_strides = strides;
    return std::move(*this);
  }

  template <typename... Ts>
  auto with_dtype(DTypeRef &&dtype) && -> TensorMatcher && {
    m_init_dtype();
    m_dtype.rebind(*dtype);
    return std::move(*this);
  }

  template <typename... Ts>
    requires(sizeof...(Ts) > 0)
  auto with_dtype() && -> TensorMatcher && {
    m_init_dtype();
    m_dtype->set_options<Ts...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  auto with_device(DeviceRef &&device) && -> TensorMatcher && {
    m_init_device();
    m_device.rebind(*device);
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
    requires(sizeof...(Codes) > 0)
  auto with_device() && -> TensorMatcher && {
    m_init_device();
    m_device->set_options<Codes...>();
    return std::move(*this);
  }

  // once we start verification, we cannot modify anymore
  auto verify(tvm::ffi::TensorView view,
              Loc_t loc = Loc_t::current()) const && -> const TensorMatcher && {
    try {
      this->m_verify_impl(view);
    } catch (PanicError &e) {
      auto oss = std::ostringstream{};
      oss << "Tensor match failed for " << this->debug_str() << " at "
          << loc.file_name() << ":" << loc.line()
          << "\n- Root cause: " << e.detail();
      throw PanicError(std::move(oss).str());
    }
    return std::move(*this);
  }

  auto debug_str() const -> std::string {
    auto oss = std::ostringstream{};
    oss << "Tensor<";
    std::size_t dim = 0;
    for (const auto &size_ref : m_shape) {
      if (dim > 0) {
        oss << ", ";
      }
      oss << size_ref->value_or_name("shape", dim++);
    }
    oss << ">";
    if (m_strides.size() > 0) {
      oss << " [strides=<";
      dim = 0;
      for (const auto &stride_ref : m_strides) {
        if (dim > 0) {
          oss << ", ";
        }
        oss << stride_ref->value_or_name("stride", dim++);
      }
      oss << ">]";
    }
    return std::move(oss).str();
  }

private:
  // WHAT: the actual checks, in order: number of dimensions; each size against its symbol or
  //       literal; strides (or contiguity if none were given); dtype; device.
  auto m_verify_impl(tvm::ffi::TensorView view) const -> void {
    const auto dim = static_cast<std::size_t>(view.dim());
    RuntimeCheck(dim == m_shape.size(), "Tensor dimension mismatch: expected ",
                 m_shape.size(), " but got ", dim);
    for (const auto i : stdv::iota(std::size_t{0}, dim)) {
      m_shape[i]->verify(view.size(i), "shape", i);
    }
    if (this->m_has_strides()) {
      for (const auto i : stdv::iota(std::size_t{0}, dim)) {
        if (view.size(i) != 1 || !m_strides[i]->has_value()) {
          // skip stride check for size 1 dimension
          m_strides[i]->verify(view.stride(i), "stride", i);
        }
      }
    } else {
      RuntimeCheck(view.is_contiguous(),
                   "Tensor is not contiguous as expected");
    }
    // since we may double verify, we will force to check
    m_dtype->verify(view.dtype());
    m_device->verify(view.device());
  }

  auto m_init_dtype() -> void {
    RuntimeCheck(!m_has_dtype, "DType already specified");
    m_has_dtype = true;
  }
  auto m_init_device() -> void {
    RuntimeCheck(!m_has_device, "Device already specified");
    m_has_device = true;
  }
  auto m_has_strides() const -> bool { return !m_strides.empty(); }

  std::span<const SizeRef> m_shape;
  std::span<const SizeRef> m_strides;
  DTypeRef m_dtype;
  DeviceRef m_device;
  bool m_has_dtype = false;
  bool m_has_device = false;
};

} // namespace host
