# =====================================================================================
# FILE MAP -- kernel/utils.py
# =====================================================================================
# WHAT THIS FILE IS: the WORKSHOP BENCH for the custom kernels: the helpers that turn a C++/CUDA
#                    source file into a callable Python function.
#
# ELI5: Writing a kernel is two jobs: (1) write the C++/CUDA, (2) get Python to call it. This file
#       does job (2). There are two ways to build a tool:
#
#         load_aot  "ahead-of-time": compile a whole .cpp/.cu file from csrc/src/ as it is.
#                   Used for tools with no tunable settings (radix compare, tensor test, NCCL).
#         load_jit  "just-in-time": paste an #include of a header from csrc/jit/ plus a one-line
#                   wrapper, with the SETTINGS baked in as C++ template arguments (for example
#                   "rows are 2048 bytes wide, use 128 threads"), then compile that exact variant.
#                   Used for index.cu and store.cu, which are tuned per tensor width.
#
#       Both go through TVM-FFI (tvm_ffi.cpp.load / load_inline), a thin bridge that lets Python call
#       a compiled function and pass torch tensors straight in (DLPack, no copy).
#
# WHY template arguments: the compiler can unroll loops and pick exact vector widths when the row
#       size is a compile-time constant, so one compiled variant per distinct width is created and
#       cached (see the functools.cache in index.py / store.py).
#
# NAMES: every module is called "minisgl__<tool>_<settings...>" so different variants never clash.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

# WHAT: where the C++/CUDA sources live: kernel/csrc/ (src/ = ahead-of-time files, jit/ = headers
#       for just-in-time variants, include/ = shared helper headers).
KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
# WHAT: compiler flags used for every build: C++20, full optimisation; CUDA files also allow
#       constexpr functions to be used on the device without annotations.
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
DEFAULT_LDFLAGS = []
# WHAT: the kinds of values that can be baked into a C++ template argument list.
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


# WHAT: a list of strings that prints as "a, b, c": exactly how C++ wants template arguments.
class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


# ELI5: the three dials every tuned kernel here exposes.
#   num_threads    how many GPU threads per block
#   max_occupancy  a hint for how many blocks may share one multiprocessor
#   use_pdl        "programmatic dependent launch": let the next kernel start preparing early
#                  (False for the kernels here)
class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    # WHAT: the dials as C++ text, e.g. "128,1,false".
    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


# WHAT: build a unique module name from the tool name and its settings.
def _make_name(*args: str) -> str:
    return "minisgl__" + "_".join(str(arg) for arg in args)


# WHAT: turn ("launch", "IndexKernel<...>::run") into the one-line C++ macro that exports that
#       function under the name "launch" so Python can call module.launch(...).
def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


# WHAT: convert Python numbers/bools to C++ literal text. bool must be tested BEFORE int
#       (in Python, True is also an int).
def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


# ELI5: "compile this file as it is, and give me a module." (Ahead-of-time flavour.)
# WHAT: the file names are looked up under csrc/src/; flags are the defaults plus any extras.
#       `build_directory=None` lets the library choose a cache folder, so a second run reuses
#       the compiled result.
def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


# ELI5: "build me THIS VARIANT of that tool." (Just-in-time flavour.)
# WHAT: instead of compiling a file directly, it writes a tiny source text: an #include of the
#       header under csrc/jit/ plus an export line per wrapper, and compiles that text. The
#       template arguments are already inside the wrapper string (see index.py / store.py).
def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
