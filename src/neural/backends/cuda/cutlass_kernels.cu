/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "neural/backends/cuda/cuda_common.h"

#include <cmath>
#include <cstdint>

#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/gemm/device/gemm.h"
// Fused MHA implementation from cutlass example #41
#include "fused_multi_head_attention/kernel_forward.h"
#include "neural/tables/activation_function.h"
#include "utils/exception.h"

namespace lczero {
namespace cudnn_backend {
namespace {

template <typename T>
struct Lc0Mish;

template <int Count>
struct Lc0Mish<cutlass::Array<float, Count>> {
  using Fragment = cutlass::Array<float, Count>;
  using Params =
      cutlass::epilogue::thread::LinearCombinationGenericParams<float>;

  CUTLASS_HOST_DEVICE
  Fragment operator()(const Fragment& input) const {
    Fragment output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Count; ++i) {
      const float x = input[i];
#if defined(__CUDA_ARCH__)
      const float e = __expf(x);
      const float n = e * e + 2.0f * e;
      const float d = __fdividef(x, n + 2.0f);
#else
      const float e = std::exp(x);
      const float n = e * e + 2.0f * e;
      const float d = x / (n + 2.0f);
#endif
      output[i] = x <= -0.6f ? n * d : x - 2.0f * d;
    }
    return output;
  }

  CUTLASS_HOST_DEVICE
  Fragment operator()(const Fragment& input, const Params&) const {
    return (*this)(input);
  }
};

template <typename T>
struct Lc0Relu2;

template <int Count>
struct Lc0Relu2<cutlass::Array<float, Count>> {
  using Fragment = cutlass::Array<float, Count>;
  using Params =
      cutlass::epilogue::thread::LinearCombinationGenericParams<float>;

  CUTLASS_HOST_DEVICE
  Fragment operator()(const Fragment& input) const {
    Fragment output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Count; ++i) {
      float x = input[i];
      if (x < 0.0f) x = 0.0f;
      output[i] = x * x;
    }
    return output;
  }

  CUTLASS_HOST_DEVICE
  Fragment operator()(const Fragment& input, const Params&) const {
    return (*this)(input);
  }
};

template <template <typename> class Activation>
using FfnEpilogue = cutlass::epilogue::thread::LinearCombinationGeneric<
    Activation, cutlass::half_t, 8, cutlass::half_t, float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling,
    cutlass::FloatRoundStyle::round_to_nearest, true>;

template <template <typename> class Activation, typename ThreadblockShape,
          typename WarpShape>
using FfnGemmConfig = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
    cutlass::layout::ColumnMajor, cutlass::half_t,
    cutlass::layout::RowMajor, cutlass::half_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 16>, FfnEpilogue<Activation>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

template <template <typename> class Activation>
using FfnGemm = FfnGemmConfig<
    Activation, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>>;

// Use four 32x64x32 warps per threadblock and an 864-block (54x16) grid for
// the exact A100 t3-distill2 FFN expansion shape.
template <template <typename> class Activation>
using FfnGemmA100T3 = FfnGemmConfig<
    Activation, cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>>;

template <typename Gemm>
bool runFfnGemmImpl(half* output, const half* input, const half* weights,
                    const half* bias, int rows, int outputs, int inputs,
                    cudaStream_t stream) {
  typename Gemm::Arguments arguments{
      {rows, outputs, inputs},
      {reinterpret_cast<const cutlass::half_t*>(input), inputs},
      {reinterpret_cast<const cutlass::half_t*>(weights), inputs},
      {reinterpret_cast<const cutlass::half_t*>(bias), 0},
      {reinterpret_cast<cutlass::half_t*>(output), outputs},
      {1.0f},
      1};

  if (Gemm::can_implement(arguments) != cutlass::Status::kSuccess) {
    return false;
  }
  Gemm gemm;
  return gemm(arguments, nullptr, stream) == cutlass::Status::kSuccess;
}

template <template <typename> class Activation>
bool runFfnGemm(half* output, const half* input, const half* weights,
                const half* bias, int rows, int outputs, int inputs,
                bool use_a100_t3_geometry, cudaStream_t stream) {
  if (use_a100_t3_geometry) {
    return runFfnGemmImpl<FfnGemmA100T3<Activation>>(
        output, input, weights, bias, rows, outputs, inputs, stream);
  }
  return runFfnGemmImpl<FfnGemm<Activation>>(
      output, input, weights, bias, rows, outputs, inputs, stream);
}

bool isAligned16(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

}  // namespace

bool fusedFfnDense1(half* output, const half* input, const half* weights,
                    const half* bias, int rows, int outputs, int inputs,
                    ActivationFunction activation, cudaStream_t stream) {
  if (rows <= 0 || outputs <= 0 || inputs <= 0 || rows % 8 != 0 ||
      outputs % 8 != 0 || inputs % 8 != 0 || !isAligned16(output) ||
      !isAligned16(input) || !isAligned16(weights) || !isAligned16(bias)) {
    return false;
  }

  int device = 0;
  int major = 0;
  int minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess ||
      major < 8) {
    return false;
  }
  const bool use_a100_t3_geometry =
      major == 8 && minor == 0 && rows == 6912 && outputs == 1024 &&
      inputs == 512;

  switch (activation) {
    case ACTIVATION_MISH:
      return runFfnGemm<Lc0Mish>(output, input, weights, bias, rows, outputs,
                                 inputs, use_a100_t3_geometry, stream);
    case ACTIVATION_RELU_2:
      return runFfnGemm<Lc0Relu2>(output, input, weights, bias, rows, outputs,
                                  inputs, use_a100_t3_geometry, stream);
    default:
      return false;
  }
}

template <bool bias>
void fusedMHACutlass(void* output, void* q, void* k, void* v, void* skip,
                     int batch_size, int num_heads, int depth,
                     cudaStream_t stream) {
  cutlass::half_t* mha_q = (cutlass::half_t*)q;
  cutlass::half_t* mha_k = (cutlass::half_t*)k;
  cutlass::half_t* mha_v = (cutlass::half_t*)v;

  constexpr int kQueriesPerBlock = 64;
  constexpr int kKeysPerBlock = 64;
  constexpr bool kSingleValueIteration = true;

  using Attention =
      AttentionKernel<cutlass::half_t,      // scalar_t
                      cutlass::arch::Sm80,  // ArchTag
                      true,                 // Memory is aligned
                      kQueriesPerBlock, kKeysPerBlock, kSingleValueIteration,
                      false,  // Supports dropout
                      bias    // Supports bias
                      >;
  static_assert(
      !Attention::kNeedsOutputAccumulatorBuffer,
      "Unhandled case in cutlass MHA: needs output accumulator buffer");

  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = mha_q;
    p.key_ptr = mha_k;
    p.value_ptr = mha_v;
    p.logsumexp_ptr = nullptr;  // Only needed for bw
    p.output_accum_ptr = nullptr;
    p.output_ptr = (cutlass::half_t*)output;
    p.attn_bias_ptr = (cutlass::half_t*)skip;

    p.scale = 1.0f / sqrt((float)depth);

    p.num_heads = num_heads;
    p.num_batches = batch_size;
    p.head_dim = depth;
    p.head_dim_value = depth;
    p.num_queries = 64;
    p.num_keys = 64;

    // All tensors are in BMHK shapes
    p.q_strideH = depth;
    p.k_strideH = depth;
    p.v_strideH = depth;
    p.q_strideM = depth * num_heads;
    p.k_strideM = depth * num_heads;
    p.v_strideM = depth * num_heads;
    p.q_strideB = p.q_strideM * 64;
    p.k_strideB = p.k_strideM * 64;
    p.v_strideB = p.v_strideM * 64;
    p.o_strideM = p.head_dim_value * p.num_heads;

    p.bias_strideH = 64 * 64;
    p.bias_strideM = 64;
    p.bias_strideB = num_heads * p.bias_strideH;
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    ReportCUDAErrors(cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  if (!Attention::check_supported(p)) {
    throw Exception("Unhandled case in cutlass MHA: check_supported failed.");
  }

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  ReportCUDAErrors(cudaGetLastError());
}

void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream) {
  if (skip == nullptr) {
    fusedMHACutlass<false>(output, mha_q, mha_k, mha_v, skip, batch_size,
                           num_heads, depth, stream);
  } else {
    fusedMHACutlass<true>(output, mha_q, mha_k, mha_v, skip, batch_size,
                          num_heads, depth, stream);
  }
}

}  // namespace cudnn_backend
}  // namespace lczero
