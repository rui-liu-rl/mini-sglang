// =====================================================================================
// FILE MAP -- kernel/csrc/include/minisgl/utils.h
// =====================================================================================
// WHAT THIS FILE IS: the shared toolbox for CPU-side C++ code in the kernel package (no GPU
//                    specifics; the GPU half is utils.cuh which includes this file).
//
//   panic / Panic / RuntimeCheck   "if this condition is false, stop with a readable error that
//                                   says which file and line"
//   PanicError                      the exception type they throw
//   div_ceil, dtype_bytes           small arithmetic helpers
//   pointer::offset                 byte-level pointer arithmetic, void* only
//
// ELI5: a safety net. Every kernel wrapper starts with RuntimeCheck(cond, "message"): when the
//       Python caller passes a tensor of the wrong kind, the error that reaches Python says
//       "Runtime check failed at <file>:<line>: <message>" instead of crashing the whole process
//       or, worse, computing garbage.
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#pragma once

// ref:
// https://forums.developer.nvidia.com/t/c-20s-source-location-compilation-error-when-using-nvcc-12-1/258026/3
// WHAT: a compatibility shim. Compiling <source_location> with nvcc 12.1 fails; these lines
//       temporarily redefine a few compiler macros just for that one include, then restore them.
//       Plain C++ compilers skip all of it (the #else branch).
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#pragma push_macro("_NODISCARD")
#pragma push_macro("__builtin_LINE")

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#define __cpp_consteval 201811L
#pragma clang diagnostic pop

#ifdef _NODISCARD
#undef _NODISCARD
#define _NODISCARD
#endif

#define consteval constexpr

#include <source_location>

#undef consteval
#pragma pop_macro("__cpp_consteval")
#pragma pop_macro("_NODISCARD")
#else
#include <source_location>
#endif

#include <dlpack/dlpack.h>

#include <concepts>
#include <ostream>
#include <sstream>
#include <utility>

namespace host {

// WHAT: the exception thrown by a failed check. `detail()` returns the message without the
//       "Runtime check failed at file:line" prefix (everything after the first ": ").
struct PanicError : public std::runtime_error {
public:
  // copy and move constructors
  PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto detail() const -> std::string_view {
    const auto sv = std::string_view{m_message};
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }

private:
  std::string m_message;
};

// WHAT: build "Runtime check failed at <file>:<line>[: message...]" and throw it. [[noreturn]]
//       tells the compiler this function never returns normally.
template <typename... Args>
[[noreturn]]
inline auto panic(std::source_location location, Args &&...args) -> void {
  std::ostringstream os;
  os << "Runtime check failed at " << location.file_name() << ":"
     << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

// WHAT: "this point must never be reached" (constructing a Panic throws immediately).
template <typename... Args> struct Panic {
  explicit Panic(Args &&...args, std::source_location location =
                                     std::source_location::current()) {
    [[unlikely]];
    ::host::panic(location, std::forward<Args>(args)...);
  }
  [[noreturn]] ~Panic() { std::terminate(); }
};

// ELI5: usage   host::RuntimeCheck(x > 0, "x must be positive, got ", x);
//       If the condition is false, throw; otherwise do nothing. The source location is captured
//       automatically as a default argument, so the message names the CALLER's line.
template <typename... Args> struct RuntimeCheck {
  template <typename T>
  explicit RuntimeCheck(
      T &&condition, Args &&...args,
      std::source_location location = std::source_location::current()) {
    if (!condition) {
      [[unlikely]];
      ::host::panic(location, std::forward<Args>(args)...);
    }
  }
};

// WHAT: deduction guides: let the compiler work out the template arguments from the call.
template <typename T, typename... Args>
explicit RuntimeCheck(T &&, Args &&...) -> RuntimeCheck<Args...>;

template <typename... Args> explicit Panic(Args &&...) -> Panic<Args...>;

// WHAT: ceiling division: div_ceil(13, 4) = 4.
template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

// WHAT: bytes per element of a DLPack dtype (bits / 8). Example: bfloat16 has 16 bits -> 2.
inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

namespace pointer {

// WHY void* only: see the note in utils.cuh. Every offset is in bytes, spelled out by the caller.
template <typename T, std::integral... U>
inline auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
inline auto offset(const T *ptr, U... offset) -> const void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

} // namespace host
