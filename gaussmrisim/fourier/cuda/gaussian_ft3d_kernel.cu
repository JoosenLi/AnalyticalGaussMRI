// Closed-form 3D Gaussian Fourier transform in PHYSICAL units, with the Fourier-CSM
// coil sum folded into the same kernel.
//
//   m_c(k) = sum_v beta_{c,v} sum_n w_n exp(-2 pi^2 (k-v)^T Sigma_n (k-v))
//                                       exp(-i 2 pi (k-v) . mu_n)
//
// Two design points that are not in the AGNC kernel this is descended from:
//
// 1. **The coil sum is accumulated in the kernel.** A block owns one (sample, mode)
//    pair, reduces over primitives to a single complex H[v,m], and then scatters
//    beta[c,v] * H into out[c,m]. The [V, M] shifted spectrum -- 8 GB at V=1000,
//    M=1e6 -- is therefore never written to memory at all: peak memory is O(C*M).
//    Materializing it and mixing with a matmul is what the PyTorch reference does,
//    and it is the binding constraint there.
//
// 2. **The phase is computed with a compensated dot product.** In millimetres,
//    k . mu is hundreds of cycles (100 um resolution over a 200 mm FOV reaches ~500),
//    and only its fractional part matters. Plain float32 leaves ~3e-5 cycles of error
//    there, which is a systematic phase error that grows with FOV and resolution.
//    An error-free transformation (fma-based TwoProduct + TwoSum) keeps the residual
//    in a second float, and the range reduction s - rintf(s) is exact by Sterbenz.
//    Double precision would also fix it, at 1/64 throughput on Ada -- this costs ~10
//    extra FP32 ops instead.
//
// Forward only, deliberately: the simulator never differentiates through the object
// spectrum. Fitting happens in the image domain and has its own kernel with a
// backward pass. If a k-space fit is ever needed, add the backward here rather than
// making this kernel carry an unused autograd path.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr float TWO_PI_SQ = 19.739208802178716f;   // 2 pi^2
constexpr float TWO_PI    = 6.283185307179586f;
constexpr int   BLOCK     = 256;

// Fractional part of (k . mu) in cycles, to ~1e-7 absolute regardless of magnitude.
__device__ __forceinline__ float phase_cycles(float kx, float ky, float kz,
                                              float mx, float my, float mz) {
    float s = 0.0f, c = 0.0f;                       // running sum and its correction
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float a = (j == 0) ? kx : (j == 1) ? ky : kz;
        const float b = (j == 0) ? mx : (j == 1) ? my : mz;
        const float p = a * b;
        const float e = __fmaf_rn(a, b, -p);        // exact product error
        const float t = s + p;                      // TwoSum of s and p
        const float lo = (fabsf(s) >= fabsf(p)) ? ((s - t) + p) : ((p - t) + s);
        s = t;
        c += lo + e;
    }
    return (s - rintf(s)) + c;                      // s - rintf(s) is exact
}

__global__ void ft_coil_forward_kernel(
    const float* __restrict__ k,        // [M, 3] cycles/mm
    const float* __restrict__ shifts,   // [V, 3] cycles/mm
    const float* __restrict__ mu,       // [N, 3] mm
    const float* __restrict__ cov6,     // [N, 6] mm^2, (xx, xy, xz, yy, yz, zz)
    const float* __restrict__ w_re,
    const float* __restrict__ w_im,     // [N]
    const float* __restrict__ beta_re,
    const float* __restrict__ beta_im,  // [C, V]
    const float* __restrict__ r2s,      // [N] 1/ms, or nullptr
    const float* __restrict__ dt,       // [M] ms since the echo, or nullptr
    int M, int N, int V, int C,
    float* __restrict__ out_re,
    float* __restrict__ out_im) {       // [C, M]
    const int m = blockIdx.x;
    const int v = blockIdx.y;
    if (m >= M) return;

    const float kx = k[m * 3 + 0] - shifts[v * 3 + 0];
    const float ky = k[m * 3 + 1] - shifts[v * 3 + 1];
    const float kz = k[m * 3 + 2] - shifts[v * 3 + 2];

    // T2* decay along the readout. It is a per-(primitive, sample) factor only because
    // every primitive belongs to exactly one tissue -- which is what lets this be one
    // launch instead of one per tissue. dt is constant across the block, so it folds
    // into the SAME exponential the Gaussian magnitude already needs: no extra
    // transcendental, no extra memory traffic beyond r2s[n].
    const float dtm = (dt == nullptr) ? 0.0f : dt[m];
    const bool decay = (r2s != nullptr) && (dt != nullptr);

    float acc_r = 0.0f, acc_i = 0.0f;
    for (int n = threadIdx.x; n < N; n += BLOCK) {
        const float cxx = cov6[n * 6 + 0], cxy = cov6[n * 6 + 1], cxz = cov6[n * 6 + 2];
        const float cyy = cov6[n * 6 + 3], cyz = cov6[n * 6 + 4], czz = cov6[n * 6 + 5];
        const float q = kx * (cxx * kx + cxy * ky + cxz * kz)
                      + ky * (cxy * kx + cyy * ky + cyz * kz)
                      + kz * (cxz * kx + cyz * ky + czz * kz);
        const float mag = decay ? __expf(-(TWO_PI_SQ * q + dtm * r2s[n]))
                                : __expf(-TWO_PI_SQ * q);
        if (mag == 0.0f) continue;                  // exact zero only; no silent cutoff
        const float fr = phase_cycles(kx, ky, kz, mu[n * 3 + 0], mu[n * 3 + 1], mu[n * 3 + 2]);
        float s, c;
        __sincosf(-TWO_PI * fr, &s, &c);
        const float br = mag * c, bi = mag * s;
        const float wr = w_re[n], wi = w_im[n];
        acc_r += br * wr - bi * wi;
        acc_i += br * wi + bi * wr;
    }

    __shared__ float sh_r[BLOCK], sh_i[BLOCK];
    sh_r[threadIdx.x] = acc_r;
    sh_i[threadIdx.x] = acc_i;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_r[threadIdx.x] += sh_r[threadIdx.x + stride];
            sh_i[threadIdx.x] += sh_i[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;

    const float hr = sh_r[0], hi = sh_i[0];
    if (beta_re == nullptr) {                       // single-channel: C == V == 1
        out_re[m] = hr;
        out_im[m] = hi;
        return;
    }
    for (int c = 0; c < C; ++c) {
        const float br = beta_re[c * V + v], bi = beta_im[c * V + v];
        atomicAdd(&out_re[c * M + m], br * hr - bi * hi);
        atomicAdd(&out_im[c * M + m], br * hi + bi * hr);
    }
}

}  // namespace

std::vector<torch::Tensor> gaussian_ft3d_forward(
    torch::Tensor k, torch::Tensor shifts, torch::Tensor mu, torch::Tensor cov6,
    torch::Tensor w_re, torch::Tensor w_im,
    c10::optional<torch::Tensor> beta_re, c10::optional<torch::Tensor> beta_im,
    c10::optional<torch::Tensor> r2s, c10::optional<torch::Tensor> dt) {
    TORCH_CHECK(k.is_cuda() && mu.is_cuda(), "gaussian_ft3d expects CUDA tensors");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "gaussian_ft3d is float32");
    const at::cuda::OptionalCUDAGuard guard(device_of(k));

    const int M = k.size(0), N = mu.size(0), V = shifts.size(0);
    const int C = beta_re.has_value() ? beta_re->size(0) : 1;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(k.device());
    auto out_re = torch::zeros({C, M}, opts);
    auto out_im = torch::zeros({C, M}, opts);

    dim3 grid(M, V);
    ft_coil_forward_kernel<<<grid, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        k.data_ptr<float>(), shifts.data_ptr<float>(), mu.data_ptr<float>(),
        cov6.data_ptr<float>(), w_re.data_ptr<float>(), w_im.data_ptr<float>(),
        beta_re.has_value() ? beta_re->data_ptr<float>() : nullptr,
        beta_im.has_value() ? beta_im->data_ptr<float>() : nullptr,
        r2s.has_value() ? r2s->data_ptr<float>() : nullptr,
        dt.has_value() ? dt->data_ptr<float>() : nullptr,
        M, N, V, C, out_re.data_ptr<float>(), out_im.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_re, out_im};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("forward", &gaussian_ft3d_forward, "3D Gaussian FT with Fourier-CSM coil sum");
}
