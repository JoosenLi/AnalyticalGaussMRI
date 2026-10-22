// Tile-binned 3D volumetric renderer for image-domain Gaussian fitting.
//
//   rho(r) = sum_n norm_n exp(-0.5 (r - mu_n)^T P_n (r - mu_n)),   P_n = Sigma_n^{-1}
//
// This is the piece AGNC does not have. Its image-domain evaluator walks every
// primitive for every point (O(P*N) with a cutoff test inside the loop), which is
// fine for the 8000 primitives a reconstruction uses and impossible for the ~1e6
// primitives and ~1e9 voxels a 100 um phantom fit needs. Here each block owns one
// 8^3 tile and only iterates the primitives whose cutoff ellipsoid reaches that tile,
// so the cost is proportional to actual overlap rather than to N.
//
// The primitive->tile assignment is built in PyTorch (see ``tiling.py``): an AABB per
// primitive, a sort by tile id, and a prefix range per tile. Keeping it there means
// the kernel receives a flat list and never has to know the parameterization -- the
// same split as the Fourier kernel, and the reason autograd can carry the quaternion
// and log-scale Jacobians without any of them entering CUDA.
//
// ``cutoff_sigma`` truncates each primitive at that Mahalanobis radius. It is an
// approximation **of the fitting objective only** and must never reach the k-space
// forward model, which is defined on the untruncated mixture. Report it.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int TILE = 8;                 // 8^3 = 512 voxels per tile
constexpr int BLOCK = 256;              // 2 voxels per thread
constexpr int BATCH = 256;              // primitives staged in shared memory

struct Prim { float mu[3], p[6], norm; };

__device__ __forceinline__ float quad(const float* p, float dx, float dy, float dz) {
    return dx * (p[0] * dx + p[1] * dy + p[2] * dz)
         + dy * (p[1] * dx + p[3] * dy + p[4] * dz)
         + dz * (p[2] * dx + p[4] * dy + p[5] * dz);
}

__global__ void render_forward_kernel(
    const float* __restrict__ mu, const float* __restrict__ prec6,
    const float* __restrict__ norm,
    const int* __restrict__ tile_start, const int* __restrict__ tile_count,
    const int* __restrict__ prim_idx,
    float ox, float oy, float oz, float dx, float dy, float dz,
    int nx, int ny, int nz, int tx, int ty, int tz, float cutoff2,
    float* __restrict__ out) {
    const int tile = blockIdx.x;
    const int ti = tile % tx, tj = (tile / tx) % ty, tk = tile / (tx * ty);
    if (tk >= tz) return;

    __shared__ float sh[BATCH * 10];
    const int start = tile_start[tile], count = tile_count[tile];

    float acc[2] = {0.0f, 0.0f};
    int vx[2], vy[2], vz[2];
    bool live[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int lv = threadIdx.x + s * BLOCK;                 // 0..511 inside the tile
        vx[s] = ti * TILE + (lv % TILE);
        vy[s] = tj * TILE + ((lv / TILE) % TILE);
        vz[s] = tk * TILE + (lv / (TILE * TILE));
        live[s] = (vx[s] < nx && vy[s] < ny && vz[s] < nz);
    }

    for (int b = 0; b < count; b += BATCH) {
        const int nb = min(BATCH, count - b);
        for (int t = threadIdx.x; t < nb; t += BLOCK) {
            const int n = prim_idx[start + b + t];
            sh[t * 10 + 0] = mu[n * 3 + 0]; sh[t * 10 + 1] = mu[n * 3 + 1];
            sh[t * 10 + 2] = mu[n * 3 + 2];
#pragma unroll
            for (int j = 0; j < 6; ++j) sh[t * 10 + 3 + j] = prec6[n * 6 + j];
            sh[t * 10 + 9] = norm[n];
        }
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            if (!live[s]) continue;
            const float rx = ox + vx[s] * dx, ry = oy + vy[s] * dy, rz = oz + vz[s] * dz;
            for (int t = 0; t < nb; ++t) {
                const float* q = &sh[t * 10];
                const float ddx = rx - q[0], ddy = ry - q[1], ddz = rz - q[2];
                const float m2 = quad(&q[3], ddx, ddy, ddz);
                if (m2 <= cutoff2) acc[s] += q[9] * __expf(-0.5f * m2);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int s = 0; s < 2; ++s)
        if (live[s]) out[(vz[s] * ny + vy[s]) * nx + vx[s]] = acc[s];
}

__global__ void render_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ mu, const float* __restrict__ prec6,
    const float* __restrict__ norm,
    const int* __restrict__ tile_start, const int* __restrict__ tile_count,
    const int* __restrict__ prim_idx,
    float ox, float oy, float oz, float dx, float dy, float dz,
    int nx, int ny, int nz, int tx, int ty, int tz, float cutoff2,
    float* __restrict__ g_mu, float* __restrict__ g_prec6, float* __restrict__ g_norm) {
    const int tile = blockIdx.x;
    const int ti = tile % tx, tj = (tile / tx) % ty, tk = tile / (tx * ty);
    if (tk >= tz) return;
    const int start = tile_start[tile], count = tile_count[tile];

    // One primitive per thread, all 512 voxels of the tile per primitive: the
    // gradient of a primitive is a sum over voxels, so accumulating it in registers
    // and doing ONE atomicAdd per (primitive, tile) is far cheaper than the
    // transposed loop, which would need 10 atomics per (primitive, voxel).
    for (int t = threadIdx.x; t < count; t += BLOCK) {
        const int n = prim_idx[start + t];
        const float mx = mu[n * 3 + 0], my = mu[n * 3 + 1], mz = mu[n * 3 + 2];
        float p[6];
#pragma unroll
        for (int j = 0; j < 6; ++j) p[j] = prec6[n * 6 + j];
        const float nn = norm[n];
        float gm[3] = {0, 0, 0}, gp[6] = {0, 0, 0, 0, 0, 0}, gn = 0.0f;

        for (int lv = 0; lv < TILE * TILE * TILE; ++lv) {
            const int vx = ti * TILE + (lv % TILE);
            const int vy = tj * TILE + ((lv / TILE) % TILE);
            const int vz = tk * TILE + (lv / (TILE * TILE));
            if (vx >= nx || vy >= ny || vz >= nz) continue;
            const float ddx = ox + vx * dx - mx, ddy = oy + vy * dy - my,
                        ddz = oz + vz * dz - mz;
            const float m2 = quad(p, ddx, ddy, ddz);
            if (m2 > cutoff2) continue;
            const float e = __expf(-0.5f * m2);
            const float g = grad_out[(vz * ny + vy) * nx + vx];
            gn += g * e;
            const float ge = g * nn * e;                       // d rho / d(exp arg)
            // d/d mu = +norm e (P d);  d/d p_jk = -0.5 norm e d_j d_k (x2 off-diagonal)
            gm[0] += ge * (p[0] * ddx + p[1] * ddy + p[2] * ddz);
            gm[1] += ge * (p[1] * ddx + p[3] * ddy + p[4] * ddz);
            gm[2] += ge * (p[2] * ddx + p[4] * ddy + p[5] * ddz);
            const float h = -0.5f * ge;
            gp[0] += h * ddx * ddx;      gp[1] += h * 2.0f * ddx * ddy;
            gp[2] += h * 2.0f * ddx * ddz; gp[3] += h * ddy * ddy;
            gp[4] += h * 2.0f * ddy * ddz; gp[5] += h * ddz * ddz;
        }
        atomicAdd(&g_norm[n], gn);
#pragma unroll
        for (int j = 0; j < 3; ++j) atomicAdd(&g_mu[n * 3 + j], gm[j]);
#pragma unroll
        for (int j = 0; j < 6; ++j) atomicAdd(&g_prec6[n * 6 + j], gp[j]);
    }
}

// Backward, v2. Two changes against v1, neither of which changes the result beyond
// float32 summation order:
//   (1) a primitive only visits the voxels of its own cutoff-ellipsoid bounding box
//       inside the tile, computed from its CURRENT precision. Every voxel outside that
//       box has m2 > cutoff2 and contributes exactly zero, so skipping it is exact. v1
//       looped all 512 voxels of the tile for every primitive.
//   (2) G consecutive lanes share one primitive and split its voxels, then reduce with
//       warp shuffles. v1 ran one thread per primitive, so a tile holding fewer than
//       BLOCK primitives left most of the block idle.
template <int G>
__global__ void render_backward_v2_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ mu, const float* __restrict__ prec6,
    const float* __restrict__ norm,
    const int* __restrict__ tile_start, const int* __restrict__ tile_count,
    const int* __restrict__ prim_idx,
    float ox, float oy, float oz, float dx, float dy, float dz,
    int nx, int ny, int nz, int tx, int ty, int tz, float cutoff2,
    float* __restrict__ g_mu, float* __restrict__ g_prec6, float* __restrict__ g_norm) {
    const int tile = blockIdx.x;
    const int ti = tile % tx, tj = (tile / tx) % ty, tk = tile / (tx * ty);
    if (tk >= tz) return;
    const int start = tile_start[tile], count = tile_count[tile];
    constexpr int PER_PASS = BLOCK / G;
    const int group = threadIdx.x / G, lane = threadIdx.x % G;
    // tile box in voxels, clipped to the volume
    const int bx0 = ti * TILE, by0 = tj * TILE, bz0 = tk * TILE;
    const int bx1 = min(bx0 + TILE, nx) - 1, by1 = min(by0 + TILE, ny) - 1,
              bz1 = min(bz0 + TILE, nz) - 1;

    for (int base = 0; base < count; base += PER_PASS) {
        const int t = base + group;
        const bool active = t < count;
        float gm[3] = {0, 0, 0}, gp[6] = {0, 0, 0, 0, 0, 0}, gn = 0.0f;
        int n = 0;
        if (active) {
            n = prim_idx[start + t];
            const float mx = mu[n * 3 + 0], my = mu[n * 3 + 1], mz = mu[n * 3 + 2];
            float p[6];
#pragma unroll
            for (int j = 0; j < 6; ++j) p[j] = prec6[n * 6 + j];
            const float nn = norm[n];
            // Sigma_jj = cofactor_jj / det(P): the half-width of the cutoff ellipsoid's
            // bounding box along axis j is cutoff * sqrt(Sigma_jj). Widened by 1% of a
            // voxel so float rounding can never drop a voxel that sits on the boundary.
            const float c00 = p[3] * p[5] - p[4] * p[4];
            const float c11 = p[0] * p[5] - p[2] * p[2];
            const float c22 = p[0] * p[3] - p[1] * p[1];
            const float det = p[0] * c00 - p[1] * (p[1] * p[5] - p[4] * p[2])
                            + p[2] * (p[1] * p[4] - p[3] * p[2]);
            const float c = sqrtf(cutoff2);
            const float hx = c * sqrtf(fmaxf(c00 / det, 0.0f)) / dx + 0.01f;
            const float hy = c * sqrtf(fmaxf(c11 / det, 0.0f)) / dy + 0.01f;
            const float hz = c * sqrtf(fmaxf(c22 / det, 0.0f)) / dz + 0.01f;
            const float cx = (mx - ox) / dx, cy = (my - oy) / dy, cz = (mz - oz) / dz;
            const int x0 = max(bx0, (int)ceilf(cx - hx)), x1 = min(bx1, (int)floorf(cx + hx));
            const int y0 = max(by0, (int)ceilf(cy - hy)), y1 = min(by1, (int)floorf(cy + hy));
            const int z0 = max(bz0, (int)ceilf(cz - hz)), z1 = min(bz1, (int)floorf(cz + hz));
            const int sx = x1 - x0 + 1, sy = y1 - y0 + 1, sz = z1 - z0 + 1;
            const int nv = (sx > 0 && sy > 0 && sz > 0) ? sx * sy * sz : 0;
            for (int v = lane; v < nv; v += G) {
                const int vx = x0 + v % sx, vy = y0 + (v / sx) % sy, vz = z0 + v / (sx * sy);
                const float ddx = ox + vx * dx - mx, ddy = oy + vy * dy - my,
                            ddz = oz + vz * dz - mz;
                const float m2 = quad(p, ddx, ddy, ddz);
                if (m2 > cutoff2) continue;
                const float e = __expf(-0.5f * m2);
                const float g = grad_out[(vz * ny + vy) * nx + vx];
                gn += g * e;
                const float ge = g * nn * e;
                gm[0] += ge * (p[0] * ddx + p[1] * ddy + p[2] * ddz);
                gm[1] += ge * (p[1] * ddx + p[3] * ddy + p[4] * ddz);
                gm[2] += ge * (p[2] * ddx + p[4] * ddy + p[5] * ddz);
                const float h = -0.5f * ge;
                gp[0] += h * ddx * ddx;        gp[1] += h * 2.0f * ddx * ddy;
                gp[2] += h * 2.0f * ddx * ddz; gp[3] += h * ddy * ddy;
                gp[4] += h * 2.0f * ddy * ddz; gp[5] += h * ddz * ddz;
            }
        }
        // reduce across the G lanes of this group; every lane of the warp takes part
#pragma unroll
        for (int off = G / 2; off > 0; off >>= 1) {
            gn += __shfl_xor_sync(0xffffffff, gn, off);
#pragma unroll
            for (int j = 0; j < 3; ++j) gm[j] += __shfl_xor_sync(0xffffffff, gm[j], off);
#pragma unroll
            for (int j = 0; j < 6; ++j) gp[j] += __shfl_xor_sync(0xffffffff, gp[j], off);
        }
        if (active && lane == 0 && gn != 0.0f) {
            atomicAdd(&g_norm[n], gn);
#pragma unroll
            for (int j = 0; j < 3; ++j) atomicAdd(&g_mu[n * 3 + j], gm[j]);
#pragma unroll
            for (int j = 0; j < 6; ++j) atomicAdd(&g_prec6[n * 6 + j], gp[j]);
        }
    }
}

}  // namespace

torch::Tensor render_forward(torch::Tensor mu, torch::Tensor prec6, torch::Tensor norm,
                             torch::Tensor tile_start, torch::Tensor tile_count,
                             torch::Tensor prim_idx, std::vector<double> origin,
                             std::vector<double> voxel, std::vector<int64_t> shape,
                             double cutoff_sigma) {
    const at::cuda::OptionalCUDAGuard guard(device_of(mu));
    const int nx = shape[0], ny = shape[1], nz = shape[2];
    const int tx = (nx + TILE - 1) / TILE, ty = (ny + TILE - 1) / TILE,
              tz = (nz + TILE - 1) / TILE;
    auto out = torch::zeros({nz, ny, nx},
                            torch::TensorOptions().dtype(torch::kFloat32).device(mu.device()));
    render_forward_kernel<<<tx * ty * tz, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        mu.data_ptr<float>(), prec6.data_ptr<float>(), norm.data_ptr<float>(),
        tile_start.data_ptr<int>(), tile_count.data_ptr<int>(), prim_idx.data_ptr<int>(),
        origin[0], origin[1], origin[2], voxel[0], voxel[1], voxel[2],
        nx, ny, nz, tx, ty, tz, cutoff_sigma * cutoff_sigma, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> render_backward_v1(
    torch::Tensor grad_out, torch::Tensor mu, torch::Tensor prec6, torch::Tensor norm,
    torch::Tensor tile_start, torch::Tensor tile_count, torch::Tensor prim_idx,
    std::vector<double> origin, std::vector<double> voxel, std::vector<int64_t> shape,
    double cutoff_sigma) {
    const at::cuda::OptionalCUDAGuard guard(device_of(mu));
    const int nx = shape[0], ny = shape[1], nz = shape[2];
    const int tx = (nx + TILE - 1) / TILE, ty = (ny + TILE - 1) / TILE,
              tz = (nz + TILE - 1) / TILE;
    auto g_mu = torch::zeros_like(mu);
    auto g_prec6 = torch::zeros_like(prec6);
    auto g_norm = torch::zeros_like(norm);
    render_backward_kernel<<<tx * ty * tz, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        grad_out.contiguous().data_ptr<float>(), mu.data_ptr<float>(),
        prec6.data_ptr<float>(), norm.data_ptr<float>(), tile_start.data_ptr<int>(),
        tile_count.data_ptr<int>(), prim_idx.data_ptr<int>(),
        origin[0], origin[1], origin[2], voxel[0], voxel[1], voxel[2],
        nx, ny, nz, tx, ty, tz, cutoff_sigma * cutoff_sigma,
        g_mu.data_ptr<float>(), g_prec6.data_ptr<float>(), g_norm.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_mu, g_prec6, g_norm};
}

std::vector<torch::Tensor> render_backward(
    torch::Tensor grad_out, torch::Tensor mu, torch::Tensor prec6, torch::Tensor norm,
    torch::Tensor tile_start, torch::Tensor tile_count, torch::Tensor prim_idx,
    std::vector<double> origin, std::vector<double> voxel, std::vector<int64_t> shape,
    double cutoff_sigma, int64_t group) {
    const at::cuda::OptionalCUDAGuard guard(device_of(mu));
    const int nx = shape[0], ny = shape[1], nz = shape[2];
    const int tx = (nx + TILE - 1) / TILE, ty = (ny + TILE - 1) / TILE,
              tz = (nz + TILE - 1) / TILE;
    auto g_mu = torch::zeros_like(mu);
    auto g_prec6 = torch::zeros_like(prec6);
    auto g_norm = torch::zeros_like(norm);
    auto go = grad_out.contiguous();
    auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH(GG) render_backward_v2_kernel<GG><<<tx * ty * tz, BLOCK, 0, stream>>>( \
        go.data_ptr<float>(), mu.data_ptr<float>(), prec6.data_ptr<float>(),          \
        norm.data_ptr<float>(), tile_start.data_ptr<int>(), tile_count.data_ptr<int>(), \
        prim_idx.data_ptr<int>(), origin[0], origin[1], origin[2],                      \
        voxel[0], voxel[1], voxel[2], nx, ny, nz, tx, ty, tz,                           \
        cutoff_sigma * cutoff_sigma, g_mu.data_ptr<float>(), g_prec6.data_ptr<float>(), \
        g_norm.data_ptr<float>())
    switch (group) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 16: LAUNCH(16); break;
        case 32: LAUNCH(32); break;
        default: LAUNCH(8); break;
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_mu, g_prec6, g_norm};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("forward", &render_forward, "tile-binned 3D Gaussian volume render");
    mod.def("backward", &render_backward, "backward, voxel sub-box + lane groups");
    mod.def("backward_v1", &render_backward_v1, "original backward (reference)");
}
