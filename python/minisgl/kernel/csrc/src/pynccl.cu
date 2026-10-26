// =====================================================================================
// FILE MAP -- kernel/csrc/src/pynccl.cu
// =====================================================================================
// WHAT THIS FILE IS: mini-sglang's own thin wrapper around NVIDIA's NCCL library, exposed to
//                    Python (kernel/pynccl.py) through TVM-FFI. It is how the GPUs of a
//                    tensor-parallel group add up partial results (all_reduce) and glue slices
//                    together (all_gather).
//
// ELI5: NCCL is the intercom between the chefs. Each chef (rank) holds a communicator object that
//       knows "I am chef r of n, and here is the shared group ticket". Two main calls:
//
//         all_reduce(t, "sum")    every rank has a tensor t of partial sums; afterwards every
//                                 rank's t holds the TOTAL (in place).
//         all_gather(dst, src)    every rank has a slice src; afterwards dst (n times bigger)
//                                 holds all slices in rank order on every rank.
//
// THE STAGING BUFFER (the one clever part):
//       At construction each rank allocates ONE big buffer of `max_bytes` with ncclMemAlloc and
//       registers it with NCCL as a "symmetric window" (NCCL_WIN_COLL_SYMMETRIC): the same
//       buffer layout exists on every GPU, which lets NCCL use faster collective algorithms.
//       all_reduce therefore works like this:
//
//         tensor fits in the buffer (size <= max_bytes):
//              copy tensor -> buffer, all-reduce the BUFFER in place, copy buffer -> tensor
//              (the copies are skipped if the tensor already lives in the buffer)
//         tensor too big:
//              all-reduce the tensor directly, in place (no staging)
//
//       all_gather never stages: it gathers straight into the destination tensor.
//
//       max_bytes is chosen in engine.py as (longest forward pass in tokens) x hidden size x bytes
//       per number, capped by the MINISGL_PYNCCL_MAX_BUFFER_SIZE setting (kernel/pynccl.py).
//
// LIMITS: only 16-bit floats are supported (float16 and bfloat16, see kNCCLDtypeMap; any other
//       dtype makes the lookup throw). Tensors must be contiguous and on a CUDA device.
//
// STREAMS: all NCCL calls are queued on the caller's current CUDA stream, so they are ordered with
//       the surrounding model kernels.
//
// Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
// =====================================================================================
#include <minisgl/nccl227.h>
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

// WHAT: the NCCL unique id is 128 bytes; it travels between processes as a list of chars.
using NCCLIDList = tvm::ffi::Array<char>;

// WHAT: turn an NCCL error code into a thrown error with NCCL's own message.
auto NCCL_CHECK(::ncclResult_t result) -> void {
  if (result != ::ncclSuccess) {
    host::RuntimeCheck(false, ::ncclGetErrorString(result));
  }
}

// WHAT: list of chars -> ncclUniqueId (checks the length first).
auto get_uid(const NCCLIDList &wrapper) -> ncclUniqueId {
  host::RuntimeCheck(wrapper.size() == NCCL_UNIQUE_ID_BYTES,
                     "Invalid NCCL ID wrapper size");
  ncclUniqueId id;
  std::copy(wrapper.begin(), wrapper.end(), id.internal);
  return id;
}

// ELI5: rank 0 calls this to make the group ticket; kernel/pynccl.py then broadcasts it.
auto create_uid() -> NCCLIDList {
  ncclUniqueId id;
  NCCL_CHECK(::ncclGetUniqueId(&id));
  return NCCLIDList(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

// WHAT: the names accepted for `op`, mapped to NCCL's reduction kinds. (The Python type hint only
//       advertises "sum"; the others are accepted here.)
const auto kNCCLReduceOPMap = std::unordered_map<std::string_view, ncclRedOp_t>{
    {"sum", ::ncclSum}, {"prod", ::ncclProd}, {"max", ::ncclMax},
    {"min", ::ncclMin}, {"avg", ::ncclAvg},
};

// WHAT: lets a DLPack dtype (code, bits, lanes) be used as a key in a hash map, by reinterpreting
//       its 32 bits as an integer.
struct DLDataTypeHash {
  auto operator()(const DLDataType &dtype) const noexcept -> std::size_t {
    return std::bit_cast<std::uint32_t>(dtype);
  }
};

// WHAT: equality for DLDataType, so the map above can compare keys.
template <typename = void>
auto operator==(const DLDataType &a, const DLDataType &b) -> bool {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

// WHAT: the dtypes NCCL calls are made with: 16-bit float and bfloat16 only.
const auto kNCCLDtypeMap =
    std::unordered_map<DLDataType, ncclDataType_t, DLDataTypeHash>{
        {{DLDataTypeCode::kDLFloat, 16, 1}, ncclFloat16},
        {{DLDataTypeCode::kDLBfloat, 16, 1}, ncclBfloat16},
    };

using std::shared_ptr;

// WHAT: small helpers so NCCL handles (raw C pointers) are released automatically when the
//       wrapper object dies: a shared_ptr that calls the matching destroy function.
template <typename T> using shared_obj = shared_ptr<std::remove_pointer_t<T>>;
template <auto Fn>
inline constexpr auto template_fn =
    [](auto &&...args) { return Fn(std::forward<decltype(args)>(args)...); };

// ELI5: one NCCLWrapper = one chef's phone line into the group.
struct NCCLWrapper : public tvm::ffi::Object {
public:
  // WHAT (in order): 1) join the group with ncclCommInitRank(comm, world size, ticket, my rank)
  //                  2) allocate the staging buffer of max_bytes
  //                  3) register it as a symmetric window
  //                  Each handle is wrapped so it is destroyed/freed/deregistered automatically.
  NCCLWrapper(int rank, int world_size, const size_t max_bytes, NCCLIDList uid)
      : m_rank(rank), m_world_size(world_size), m_max_bytes(max_bytes) {
    ncclUniqueId id = get_uid(uid);
    ncclComm_t comm;
    NCCL_CHECK(::ncclCommInitRank(&comm, m_world_size, id, m_rank));
    m_comm = {comm, template_fn<::ncclCommDestroy>};

    void *buf;
    NCCL_CHECK(::ncclMemAlloc(&buf, max_bytes));
    m_sym_mem = {buf, template_fn<::ncclMemFree>};

    ncclWindow_t win;
    NCCL_CHECK(::ncclCommWindowRegister(comm, buf, max_bytes, &win,
                                        NCCL_WIN_COLL_SYMMETRIC));
    m_win = {win, [comm = m_comm](ncclWindow_t w) {
               return NCCL_CHECK(::ncclCommWindowDeregister(comm.get(), w));
             }};
  }

  // ELI5: "add up everyone's tensor and give everyone the total, in place."
  // WHAT: checks the tensor (CUDA, contiguous), looks up the NCCL dtype and reduction, finds the
  //       current CUDA stream of the tensor's device, then either goes through the staging buffer
  //       or works directly on the tensor (see the header comment).
  // SHAPE: any contiguous 16-bit float tensor; count = number of elements; bytes = count x 2.
  auto all_reduce(tvm::ffi::TensorView t, std::string op) const -> void {
    using namespace host;
    RuntimeCheck(t.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(t.is_contiguous(), "Tensor must be contiguous");
    const auto size_dim = static_cast<size_t>(t.shape().Product());
    const auto dtype = kNCCLDtypeMap.at(t.dtype());
    const auto size_bytes = size_dim * (t.dtype().bits / 8);
    const auto data_ptr = t.data_ptr();
    const auto reduce_op = kNCCLReduceOPMap.at(op);
    const auto stream = LaunchKernel::resolve_device(t.device());

    if (size_bytes <= m_max_bytes) { // use internal buffer
      const auto buf_ptr = m_sym_mem.get();
      // WHAT: skip the copies if the tensor is already the buffer itself.
      const auto need_memcpy = (buf_ptr != data_ptr);
      if (need_memcpy) {
        CUDA_CHECK(::cudaMemcpyAsync(buf_ptr, data_ptr, size_bytes,
                                     ::cudaMemcpyDeviceToDevice, stream));
      }
      NCCL_CHECK(::ncclAllReduce(
          /*sendbuff=*/buf_ptr,
          /*recvbuff=*/buf_ptr,
          /*count=*/size_dim,
          /*datatype=*/dtype,
          /*op=*/reduce_op,
          /*comm=*/m_comm.get(),
          /*stream=*/stream));
      if (need_memcpy) {
        CUDA_CHECK(::cudaMemcpyAsync(data_ptr, buf_ptr, size_bytes,
                                     ::cudaMemcpyDeviceToDevice, stream));
      }
    } else {
      // WHAT: too big for the buffer: reduce in place on the tensor itself.
      NCCL_CHECK(::ncclAllReduce(
          /*sendbuff=*/data_ptr,
          /*recvbuff=*/data_ptr,
          /*count=*/size_dim,
          /*datatype=*/dtype,
          /*op=*/reduce_op,
          /*comm=*/m_comm.get(),
          /*stream=*/stream));
    }
  }

  // ELI5: "everyone hands in a slice; everyone receives all slices laid end to end."
  // SHAPE: src [m, ...] on each rank; dst [m * world_size, ...] (dim 0 is checked); rank r's slice
  //        lands at dst[r*m : (r+1)*m]. Used by ParallelLMHead to assemble the full vocabulary.
  auto all_gather(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) const
      -> void {
    using namespace host;
    RuntimeCheck(src.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(src.is_contiguous(), "Tensor must be contiguous");
    RuntimeCheck(dst.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(dst.is_contiguous(), "Tensor must be contiguous");
    RuntimeCheck(dst.size(0) == src.size(0) * m_world_size,
                 "Destination tensor has incorrect size");
    const auto size_dim = static_cast<size_t>(src.shape().Product());
    const auto dtype = kNCCLDtypeMap.at(src.dtype());
    const auto src_ptr = src.data_ptr();
    const auto dst_ptr = dst.data_ptr();
    const auto stream = LaunchKernel::resolve_device(src.device());
    // do not use internal buffer for all_gather, directly gather to output
    // tensor
    NCCL_CHECK(::ncclAllGather(
        /*sendbuff=*/src_ptr,
        /*recvbuff=*/dst_ptr,
        /*sendcount=*/size_dim,
        /*datatype=*/dtype,
        /*comm=*/m_comm.get(),
        /*stream=*/stream));
  }

  // WHAT: the address of the staging buffer (a tensor placed there would skip the copies).
  auto get_buffer() const -> void * { return m_sym_mem.get(); }

  // WHAT: registers the class with TVM-FFI under the name Python uses ("minisgl.NCCLWrapper").
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("minisgl.NCCLWrapper", NCCLWrapper,
                                    tvm::ffi::Object);

private:
  int m_rank;
  int m_world_size;
  size_t m_max_bytes;
  shared_obj<ncclComm_t> m_comm;
  shared_ptr<void> m_sym_mem;
  shared_obj<ncclWindow_t> m_win;
};

// WHAT: runs once when the library loads: tells TVM-FFI which constructor and methods Python may
//       call on a NCCLWrapper (init, all_reduce, all_gather, get_buffer).
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<NCCLWrapper>()
      .def(refl::init<int, int, size_t, NCCLIDList>(), "__init__")
      .def("all_reduce", &NCCLWrapper::all_reduce)
      .def("all_gather", &NCCLWrapper::all_gather)
      .def("get_buffer", &NCCLWrapper::get_buffer);
}

// WHAT: also export a plain function module.create_nccl_uid() for rank 0.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_nccl_uid, &create_uid);

} // namespace
