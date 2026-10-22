// Analytic gradient of a real Gaussian mixture at continuous points, for the
// Monte-Carlo total-variation regularizer of the image-domain fit
// (gaussmrisim/fitting/continuous_tv.py). Ported from the ICLR demo's
// recon/cuda/gaussian_reg3d_kernel.cu, restricted to a REAL object.
//
//   grad rho(x_p) = - sum_n c_n K_n (x_p - mu_n) exp(-0.5 (x_p-mu_n)^T K_n (x_p-mu_n))
//
// The kernel takes geometry, not parameters: mu, K = Sigma^-1 (packed symmetric
// [xx, xy, xz, yy, yz, zz]) and the PEAK coefficient c = m / ((2 pi)^{3/2} |Sigma|^{1/2}).
// Building those from (log_scale, quaternion, mass) stays in PyTorch, where autograd
// owns the Jacobians and the integrated-mass convention lives in one place.
//
// This is a regularizer of the FIT, not part of the k-space object: it is O(M N) with
// no cutoff, and no tolerance is claimed for it beyond matching the PyTorch reference.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace {

constexpr int PT_THREADS = 64;
constexpr int RED_THREADS = 256;

__device__ __forceinline__ void sym_load(const float* __restrict__ a, int n, float m[6]) {
#pragma unroll
  for (int i = 0; i < 6; ++i) m[i] = a[6 * n + i];
}

__device__ __forceinline__ void sym_mv(const float m[6], const float x[3], float y[3]) {
  y[0] = m[0] * x[0] + m[1] * x[1] + m[2] * x[2];
  y[1] = m[1] * x[0] + m[3] * x[1] + m[4] * x[2];
  y[2] = m[2] * x[0] + m[4] * x[1] + m[5] * x[2];
}

__device__ __forceinline__ void block_reduce(float* sh, int cnt) {
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      for (int j = 0; j < cnt; ++j)
        sh[threadIdx.x * cnt + j] += sh[(threadIdx.x + stride) * cnt + j];
    __syncthreads();
  }
}

// 2D grid: x over point tiles, y over primitive chunks, atomics into the output.
__global__ void grad_rho_forward_kernel(
    const float* __restrict__ pts, const float* __restrict__ mu,
    const float* __restrict__ kinv, const float* __restrict__ c,
    int M, int N, int chunk, float* __restrict__ g) {
  const int p = blockIdx.x * PT_THREADS + threadIdx.x;
  const int n0 = blockIdx.y * chunk;
  const int n1 = min(n0 + chunk, N);
  if (p >= M) return;
  const float x[3] = {pts[3 * p], pts[3 * p + 1], pts[3 * p + 2]};
  float a[3] = {0.f, 0.f, 0.f};
  for (int n = n0; n < n1; ++n) {
    const float d[3] = {x[0] - mu[3 * n], x[1] - mu[3 * n + 1], x[2] - mu[3 * n + 2]};
    float k[6]; sym_load(kinv, n, k);
    float z[3]; sym_mv(k, d, z);
    const float q = d[0] * z[0] + d[1] * z[1] + d[2] * z[2];
    if (q > 80.f) continue;                 // exp(-40) ~ 4e-18: exactly 0 in float32
    const float w = c[n] * expf(-0.5f * q);
#pragma unroll
    for (int i = 0; i < 3; ++i) a[i] -= w * z[i];
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) atomicAdd(&g[3 * p + i], a[i]);
}

// One block per primitive, threads stream over points; recomputes the forward.
//   d/dc     = -b (t.z)
//   d/dmu_k  = c b [ (K t)_k - z_k (t.z) ]
//   d/dK_ab  = c b [ 0.5 d_a d_b (t.z) - sym(t_a d_b) ]
__global__ void grad_rho_backward_kernel(
    const float* __restrict__ gp, const float* __restrict__ pts,
    const float* __restrict__ mu, const float* __restrict__ kinv,
    const float* __restrict__ c, int M, int N,
    float* __restrict__ gmu, float* __restrict__ gkinv, float* __restrict__ gc) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const float m[3] = {mu[3 * n], mu[3 * n + 1], mu[3 * n + 2]};
  float k[6]; sym_load(kinv, n, k);
  const float cn = c[n];
  float a[10];                               // 3 mu, 6 kinv, 1 c
#pragma unroll
  for (int j = 0; j < 10; ++j) a[j] = 0.f;
  for (int p = threadIdx.x; p < M; p += blockDim.x) {
    const float d[3] = {pts[3 * p] - m[0], pts[3 * p + 1] - m[1], pts[3 * p + 2] - m[2]};
    float z[3]; sym_mv(k, d, z);
    const float q = d[0] * z[0] + d[1] * z[1] + d[2] * z[2];
    if (q > 80.f) continue;
    const float b = expf(-0.5f * q);
    const float t[3] = {gp[3 * p], gp[3 * p + 1], gp[3 * p + 2]};
    const float tz = t[0] * z[0] + t[1] * z[1] + t[2] * z[2];
    a[9] -= b * tz;
    const float cb = cn * b;
    float kt[3]; sym_mv(k, t, kt);
#pragma unroll
    for (int i = 0; i < 3; ++i) a[i] += cb * (kt[i] - z[i] * tz);
    const float h = 0.5f * cb * tz;
    a[3] += h * d[0] * d[0] - cb * t[0] * d[0];
    a[4] += 2.f * h * d[0] * d[1] - cb * (t[0] * d[1] + t[1] * d[0]);
    a[5] += 2.f * h * d[0] * d[2] - cb * (t[0] * d[2] + t[2] * d[0]);
    a[6] += h * d[1] * d[1] - cb * t[1] * d[1];
    a[7] += 2.f * h * d[1] * d[2] - cb * (t[1] * d[2] + t[2] * d[1]);
    a[8] += h * d[2] * d[2] - cb * t[2] * d[2];
  }
  __shared__ float sh[RED_THREADS * 10];
#pragma unroll
  for (int j = 0; j < 10; ++j) sh[threadIdx.x * 10 + j] = a[j];
  block_reduce(sh, 10);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 3; ++i) gmu[3 * n + i] = sh[i];
#pragma unroll
    for (int i = 0; i < 6; ++i) gkinv[6 * n + i] = sh[3 + i];
    gc[n] = sh[9];
  }
}

}  // namespace

torch::Tensor grad_rho_forward(torch::Tensor pts, torch::Tensor mu, torch::Tensor kinv,
                               torch::Tensor c, int64_t chunk) {
  const at::cuda::CUDAGuard guard(pts.device());
  const int M = pts.size(0), N = mu.size(0);
  auto g = torch::zeros({M, 3}, pts.options());
  if (M == 0 || N == 0) return g;
  const int ch = std::max<int>(1, static_cast<int>(chunk));
  dim3 grid((M + PT_THREADS - 1) / PT_THREADS, (N + ch - 1) / ch);
  grad_rho_forward_kernel<<<grid, PT_THREADS>>>(
      pts.data_ptr<float>(), mu.data_ptr<float>(), kinv.data_ptr<float>(),
      c.data_ptr<float>(), M, N, ch, g.data_ptr<float>());
  return g;
}

std::vector<torch::Tensor> grad_rho_backward(torch::Tensor gp, torch::Tensor pts,
                                             torch::Tensor mu, torch::Tensor kinv,
                                             torch::Tensor c) {
  const at::cuda::CUDAGuard guard(pts.device());
  const int M = pts.size(0), N = mu.size(0);
  auto opts = pts.options();
  auto gmu = torch::zeros({N, 3}, opts), gkinv = torch::zeros({N, 6}, opts);
  auto gc = torch::zeros({N}, opts);
  if (M == 0 || N == 0) return {gmu, gkinv, gc};
  grad_rho_backward_kernel<<<N, RED_THREADS>>>(
      gp.data_ptr<float>(), pts.data_ptr<float>(), mu.data_ptr<float>(),
      kinv.data_ptr<float>(), c.data_ptr<float>(), M, N, gmu.data_ptr<float>(),
      gkinv.data_ptr<float>(), gc.data_ptr<float>());
  return {gmu, gkinv, gc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grad_rho_forward", &grad_rho_forward, "analytic grad rho (CUDA)");
  m.def("grad_rho_backward", &grad_rho_backward, "grad rho backward (CUDA)");
}
