# =====================================================================================
# FILE MAP -- kernel/__main__.py
# =====================================================================================
# WHAT THIS FILE IS: a developer convenience script, run with `python -m minisgl.kernel`.
#                    It writes a `.clangd` file so that editors can understand the CUDA/C++ sources
#                    in kernel/csrc (include paths, GPU architecture, language standard).
#
# ELI5: Editors can't guess where the TVM-FFI and DLPack headers live or which GPU you have, so the
#       C++ files show red squiggles. This script finds the header folders, asks nvidia-smi for
#       your GPU's compute capability (e.g. "9.0" -> sm_90) and writes the matching `.clangd`.
#       It does not touch any runtime behaviour of the engine. Requires an NVIDIA GPU + nvidia-smi.
#
# BEHAVIOUR: if a `.clangd` already exists in the current folder it is NOT overwritten; the
#       suggested content is printed instead.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
# WHAT: refuse to be imported; this file is only meant to be run as a script.
assert __name__ == "__main__"


def generate_clangd():
    import os
    import subprocess

    from minisgl.kernel.utils import DEFAULT_INCLUDE
    from minisgl.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    # WHAT: ask the driver for the GPU's compute capability; the first line is GPU 0.
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")
    # WHAT: treat the files as CUDA ("-xcuda") for that GPU, C++20, with warnings on; the header
    #       folders are added as "system" includes so warnings inside them are suppressed.
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-std=c++20",
            "-Wall",
            "-Wextra",
        ]
        + [f"-isystem{path}" for path in include_paths]
    )
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd"):
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


generate_clangd()
