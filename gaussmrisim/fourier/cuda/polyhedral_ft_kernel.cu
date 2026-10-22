// Exact Fourier transform of closed triangle meshes (Ngo et al., MRM 2016).
// See gaussmrisim/fourier/polyhedral_ft.py for the formula and record layout.
//
// One thread per k-space sample; face records are staged through shared memory in
// batches and every thread reduces over all faces, the same layout as the Gaussian FT
// kernel so the two can be timed against each other on equal terms. T is the type of
// the face arithmetic (float or double); the phase k.c is always formed in double and
// reduced mod 1 cycle before it is cast.
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int REC = 35;
constexpr int BLOCK = 128;
constexpr int BATCH = 64;
constexpr double PI = 3.14159265358979323846;

template <typename T> __device__ __forceinline__ void sc2pi(double cyc, T* s, T* c);
template <> __device__ __forceinline__ void sc2pi<double>(double cyc, double* s, double* c) {
    sincospi(2.0 * cyc, s, c);
}
template <> __device__ __forceinline__ void sc2pi<float>(double cyc, float* s, float* c) {
    sincospif(2.0f * (float)cyc, s, c);
}
template <typename T> __device__ __forceinline__ T sinc_(T x);
template <> __device__ __forceinline__ double sinc_<double>(double x) {
    return fabs(x) < 1e-8 ? 1.0 - (PI * PI / 6.0) * x * x : sinpi(x) / (PI * x);
}
template <> __device__ __forceinline__ float sinc_<float>(float x) {
    return fabsf(x) < 1e-4f ? 1.0f - 1.6449341f * x * x : sinpif(x) / (3.14159265f * x);
}

template <typename T>
__global__ void poly_ft_kernel(const double* __restrict__ rec, int F,
                               const double* __restrict__ kv, int M,
                               double par_tol, double* __restrict__ out) {
    __shared__ double sh[BATCH * REC];
    const int m = blockIdx.x * BLOCK + threadIdx.x;
    const bool live = m < M;
    double kxd = 0, kyd = 0, kzd = 0;
    if (live) { kxd = kv[3 * m]; kyd = kv[3 * m + 1]; kzd = kv[3 * m + 2]; }
    const T kx = (T)kxd, ky = (T)kyd, kz = (T)kzd;
    const T kk = kx * kx + ky * ky + kz * kz;
    T re = 0, im = 0;

    for (int b = 0; b < F; b += BATCH) {
        const int nb = min(BATCH, F - b);
        for (int i = threadIdx.x; i < nb * REC; i += BLOCK) sh[i] = rec[(size_t)b * REC + i];
        __syncthreads();
        if (live && kk > (T)0) {
            for (int f = 0; f < nb; ++f) {
                const double* r = &sh[f * REC];
                const T kN = kx * (T)r[0] + ky * (T)r[1] + kz * (T)r[2];
                const T den = kk - kN * kN;
                const T val = (T)r[34];
                if (den <= (T)par_tol * kk) {
                    // k parallel to the face normal
                    const double cyc = kxd * r[30] + kyd * r[31] + kzd * r[32];
                    T s, c;
                    sc2pi<T>(cyc - floor(cyc), &s, &c);
                    // -i pi kN 2A (c - i s) = -pi kN 2A (s + i c)
                    const T a = (T)PI * kN * (T)r[33] * val;
                    re += -a * s;
                    im += -a * c;
                    continue;
                }
                T sre = 0, sim = 0;
#pragma unroll
                for (int e = 0; e < 3; ++e) {
                    const double* q = r + 3 + 9 * e;
                    const T kn = kx * (T)q[0] + ky * (T)q[1] + kz * (T)q[2];
                    const T kt = kx * (T)q[3] + ky * (T)q[4] + kz * (T)q[5];
                    const double cyc = kxd * q[6] + kyd * q[7] + kzd * q[8];
                    T s, c;
                    sc2pi<T>(cyc - floor(cyc), &s, &c);
                    const T w = kn * sinc_<T>(kt);
                    sre += w * c;
                    sim -= w * s;
                }
                const T g = val * kN / den;
                re += g * sre;
                im += g * sim;
            }
        }
        __syncthreads();
    }
    if (live) {
        const double s = -1.0 / ((2.0 * PI) * (2.0 * PI) * (double)kk);
        out[2 * m] = kk > (T)0 ? s * (double)re : 0.0;
        out[2 * m + 1] = kk > (T)0 ? s * (double)im : 0.0;
    }
}

// Fractional part of (k . c) in cycles with a compensated float32 dot product -- the
// same routine as the Gaussian FT kernel, so the fp32 timing comparison is like for like.
__device__ __forceinline__ float phase_cycles(float kx, float ky, float kz,
                                              float mx, float my, float mz) {
    float s = 0.0f, c = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const float a = (j == 0) ? kx : (j == 1) ? ky : kz;
        const float b = (j == 0) ? mx : (j == 1) ? my : mz;
        const float p = a * b;
        const float e = __fmaf_rn(a, b, -p);
        const float t = s + p;
        const float lo = (fabsf(s) >= fabsf(p)) ? ((s - t) + p) : ((p - t) + s);
        s = t;
        c += lo + e;
    }
    return (s - rintf(s)) + c;
}

// All-float32 variant: float records in shared memory, compensated fp32 phase.
__global__ void poly_ft_kernel_f32(const float* __restrict__ rec, int F,
                                   const float* __restrict__ kv, int M,
                                   float par_tol, float near_par,
                                   double* __restrict__ out) {
    __shared__ float sh[BATCH * 2 * REC];
    constexpr int B2 = BATCH * 2;
    const int m = blockIdx.x * BLOCK + threadIdx.x;
    const bool live = m < M;
    float kx = 0, ky = 0, kz = 0;
    if (live) { kx = kv[3 * m]; ky = kv[3 * m + 1]; kz = kv[3 * m + 2]; }
    const float kk = kx * kx + ky * ky + kz * kz;
    float re = 0, im = 0;
    for (int b = 0; b < F; b += B2) {
        const int nb = min(B2, F - b);
        for (int i = threadIdx.x; i < nb * REC; i += BLOCK) sh[i] = rec[(size_t)b * REC + i];
        __syncthreads();
        if (live && kk > 0.0f) {
            for (int f = 0; f < nb; ++f) {
                const float* r = &sh[f * REC];
                const float kN = kx * r[0] + ky * r[1] + kz * r[2];
                const float den = kk - kN * kN;
                const float val = r[34];
                if (den <= par_tol * kk) {
                    float sn, cs;
                    sincospif(2.0f * phase_cycles(kx, ky, kz, r[30], r[31], r[32]), &sn, &cs);
                    const float a = 3.14159265f * kN * r[33] * val;
                    re += -a * sn;
                    im += -a * cs;
                    continue;
                }
                if (den < near_par * kk) {
                    // k within ~6 degrees of the face normal: the edge sum nearly cancels
                    // before the division by den, which float32 cannot carry (measured
                    // 7.8e-4 of the volume on a box). These are ~1% of face-k pairs;
                    // evaluate them in float64. The phase stays the compensated fp32
                    // value, which is already exact to ~1e-7 cycles.
                    const double dkx = kx, dky = ky, dkz = kz;
                    const double dkN = dkx * r[0] + dky * r[1] + dkz * r[2];
                    const double dden = dkx * dkx + dky * dky + dkz * dkz - dkN * dkN;
                    double dre = 0, dim = 0;
#pragma unroll
                    for (int e = 0; e < 3; ++e) {
                        const float* q = r + 3 + 9 * e;
                        const double kn = dkx * q[0] + dky * q[1] + dkz * q[2];
                        const double kt = dkx * q[3] + dky * q[4] + dkz * q[5];
                        double sn, cs;
                        sincospi(2.0 * (double)phase_cycles(kx, ky, kz, q[6], q[7], q[8]),
                                 &sn, &cs);
                        const double w = kn * sinc_<double>(kt);
                        dre += w * cs;
                        dim -= w * sn;
                    }
                    const double g = (double)val * dkN / dden;
                    re += (float)(g * dre);
                    im += (float)(g * dim);
                    continue;
                }
                float sre = 0, sim = 0;
#pragma unroll
                for (int e = 0; e < 3; ++e) {
                    const float* q = r + 3 + 9 * e;
                    const float kn = kx * q[0] + ky * q[1] + kz * q[2];
                    const float kt = kx * q[3] + ky * q[4] + kz * q[5];
                    float sn, cs;
                    sincospif(2.0f * phase_cycles(kx, ky, kz, q[6], q[7], q[8]), &sn, &cs);
                    const float w = kn * sinc_<float>(kt);
                    sre += w * cs;
                    sim -= w * sn;
                }
                const float g = val * kN / den;
                re += g * sre;
                im += g * sim;
            }
        }
        __syncthreads();
    }
    if (live) {
        const double s = -1.0 / ((2.0 * PI) * (2.0 * PI) * (double)kk);
        out[2 * m] = kk > 0.0f ? s * (double)re : 0.0;
        out[2 * m + 1] = kk > 0.0f ? s * (double)im : 0.0;
    }
}

template <typename T>
torch::Tensor forward(torch::Tensor rec, torch::Tensor k, double par_tol) {
    const at::cuda::OptionalCUDAGuard guard(device_of(rec));
    TORCH_CHECK(rec.dtype() == torch::kFloat64 && k.dtype() == torch::kFloat64);
    TORCH_CHECK(rec.size(1) == REC, "face record must have 35 doubles");
    const int F = rec.size(0), M = k.size(0);
    auto out = torch::empty({M, 2}, rec.options());
    poly_ft_kernel<T><<<(M + BLOCK - 1) / BLOCK, BLOCK, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        rec.data_ptr<double>(), F, k.data_ptr<double>(), M, par_tol, out.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---------------------------------------------------------------------------------
// Edge form. Any union of weighted planar polygons with consistent orientation is a
// sum over oriented edges: edge e on a plane with unit normal N, weight w (the value of
// the polygon it bounds), in-plane outward normal n, direction t = n-perp, length L,
// midpoint c, and s = N . ((a - o) x (b - o)) its shoelace term about a point o of its
// plane (the k || N limit needs the area, and sum_e s_e = 2A per region). Record:
//   N(3) | L n (3) | L t (3) | c (3) | s | w        (14 values)
// This is what lets a tissue cut into cells carry its caps without assembling them
// into closed polygons: each cap is just more edges.
constexpr int EREC = 14;

template <typename T>
__global__ void edge_ft_kernel(const double* __restrict__ rec, int E,
                               const double* __restrict__ kv, int M,
                               double* __restrict__ out) {
    __shared__ double sh[BATCH * 2 * EREC];
    constexpr int B2 = BATCH * 2;
    const int m = blockIdx.x * BLOCK + threadIdx.x;
    const bool live = m < M;
    double kxd = 0, kyd = 0, kzd = 0;
    if (live) { kxd = kv[3 * m]; kyd = kv[3 * m + 1]; kzd = kv[3 * m + 2]; }
    const double kkd = kxd * kxd + kyd * kyd + kzd * kzd;
    const T kx = (T)kxd, ky = (T)kyd, kz = (T)kzd, kk = (T)kkd;
    T re = 0, im = 0;
    for (int b = 0; b < E; b += B2) {
        const int nb = min(B2, E - b);
        for (int i = threadIdx.x; i < nb * EREC; i += BLOCK) sh[i] = rec[(size_t)b * EREC + i];
        __syncthreads();
        if (live && kkd > 0) {
            for (int f = 0; f < nb; ++f) {
                const double* r = &sh[f * EREC];
                const T kN = kx * (T)r[0] + ky * (T)r[1] + kz * (T)r[2];
                const T den = kk - kN * kN;
                const double cyc = kxd * r[9] + kyd * r[10] + kzd * r[11];
                if (den <= (T)1e-12 * kk) {
                    T sn, cs;
                    sc2pi<T>(cyc - floor(cyc), &sn, &cs);
                    const T a = (T)PI * kN * (T)r[13] * (T)r[12];   // pi kN w s
                    re += -a * sn;
                    im += -a * cs;
                    continue;
                }
                if (sizeof(T) == 4 && den < (T)1e-2 * kk) {
                    // near-parallel: the per-face sum cancels; evaluate in float64
                    const double dkN = kxd * r[0] + kyd * r[1] + kzd * r[2];
                    const double dden = kkd - dkN * dkN;
                    const double kn = kxd * r[3] + kyd * r[4] + kzd * r[5];
                    const double kt = kxd * r[6] + kyd * r[7] + kzd * r[8];
                    double sn, cs;
                    sincospi(2.0 * (cyc - floor(cyc)), &sn, &cs);
                    const double g = r[13] * dkN / dden * kn * sinc_<double>(kt);
                    re += (T)(g * cs);
                    im += (T)(-g * sn);
                    continue;
                }
                const T kn = kx * (T)r[3] + ky * (T)r[4] + kz * (T)r[5];
                const T kt = kx * (T)r[6] + ky * (T)r[7] + kz * (T)r[8];
                T sn, cs;
                sc2pi<T>(cyc - floor(cyc), &sn, &cs);
                const T g = (T)r[13] * kN / den * kn * sinc_<T>(kt);
                re += g * cs;
                im -= g * sn;
            }
        }
        __syncthreads();
    }
    if (live) {
        const double sc = -1.0 / ((2.0 * PI) * (2.0 * PI) * kkd);
        out[2 * m] = kkd > 0 ? sc * (double)re : 0.0;
        out[2 * m + 1] = kkd > 0 ? sc * (double)im : 0.0;
    }
}

// All-float32 edge kernel: float records, compensated fp32 phase (phase_cycles), the
// near-parallel fallback in float64 -- the edge counterpart of poly_ft_kernel_f32.
__global__ void edge_ft_kernel_f32(const float* __restrict__ rec, int E,
                                   const float* __restrict__ kv, int M,
                                   double* __restrict__ out) {
    __shared__ float sh[BATCH * 4 * EREC];
    constexpr int B4 = BATCH * 4;
    const int m = blockIdx.x * BLOCK + threadIdx.x;
    const bool live = m < M;
    float kx = 0, ky = 0, kz = 0;
    if (live) { kx = kv[3 * m]; ky = kv[3 * m + 1]; kz = kv[3 * m + 2]; }
    const float kk = kx * kx + ky * ky + kz * kz;
    float re = 0, im = 0;
    for (int b = 0; b < E; b += B4) {
        const int nb = min(B4, E - b);
        for (int i = threadIdx.x; i < nb * EREC; i += BLOCK) sh[i] = rec[(size_t)b * EREC + i];
        __syncthreads();
        if (live && kk > 0.0f) {
            for (int f = 0; f < nb; ++f) {
                const float* r = &sh[f * EREC];
                const float kN = kx * r[0] + ky * r[1] + kz * r[2];
                const float den = kk - kN * kN;
                const float cyc = phase_cycles(kx, ky, kz, r[9], r[10], r[11]);
                float sn, cs;
                if (den <= 1e-12f * kk) {
                    sincospif(2.0f * cyc, &sn, &cs);
                    const float a = 3.14159265f * kN * r[13] * r[12];
                    re += -a * sn;
                    im += -a * cs;
                    continue;
                }
                if (den < 1e-2f * kk) {
                    const double dkx = kx, dky = ky, dkz = kz;
                    const double dkN = dkx * r[0] + dky * r[1] + dkz * r[2];
                    const double dden = dkx * dkx + dky * dky + dkz * dkz - dkN * dkN;
                    const double kn = dkx * r[3] + dky * r[4] + dkz * r[5];
                    const double kt = dkx * r[6] + dky * r[7] + dkz * r[8];
                    double dsn, dcs;
                    sincospi(2.0 * (double)cyc, &dsn, &dcs);
                    const double g = (double)r[13] * dkN / dden * kn * sinc_<double>(kt);
                    re += (float)(g * dcs);
                    im += (float)(-g * dsn);
                    continue;
                }
                const float kn = kx * r[3] + ky * r[4] + kz * r[5];
                const float kt = kx * r[6] + ky * r[7] + kz * r[8];
                sincospif(2.0f * cyc, &sn, &cs);
                const float g = r[13] * kN / den * kn * sinc_<float>(kt);
                re += g * cs;
                im -= g * sn;
            }
        }
        __syncthreads();
    }
    if (live) {
        const double sc = -1.0 / ((2.0 * PI) * (2.0 * PI) * (double)kk);
        out[2 * m] = kk > 0.0f ? sc * (double)re : 0.0;
        out[2 * m + 1] = kk > 0.0f ? sc * (double)im : 0.0;
    }
}

template <typename T>
torch::Tensor forward_edges(torch::Tensor rec, torch::Tensor k) {
    const at::cuda::OptionalCUDAGuard guard(device_of(rec));
    TORCH_CHECK(rec.dtype() == torch::kFloat64 && k.dtype() == torch::kFloat64);
    TORCH_CHECK(rec.size(1) == EREC, "edge record must have 14 doubles");
    const int E = rec.size(0), M = k.size(0);
    auto out = torch::empty({M, 2}, rec.options());
    edge_ft_kernel<T><<<(M + BLOCK - 1) / BLOCK, BLOCK, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        rec.data_ptr<double>(), E, k.data_ptr<double>(), M, out.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

torch::Tensor forward_double(torch::Tensor rec, torch::Tensor k) {
    return forward<double>(rec, k, 1e-14);
}

torch::Tensor forward_f32(torch::Tensor rec, torch::Tensor k) {
    const at::cuda::OptionalCUDAGuard guard(device_of(rec));
    TORCH_CHECK(rec.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32);
    const int F = rec.size(0), M = k.size(0);
    auto out = torch::empty({M, 2}, rec.options().dtype(torch::kFloat64));
    poly_ft_kernel_f32<<<(M + BLOCK - 1) / BLOCK, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        rec.data_ptr<float>(), F, k.data_ptr<float>(), M, 1e-7f, 1e-2f, out.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor forward_edges_double(torch::Tensor rec, torch::Tensor k) {
    return forward_edges<double>(rec, k);
}
torch::Tensor forward_edges_float(torch::Tensor rec, torch::Tensor k) {
    return forward_edges<float>(rec, k);
}

torch::Tensor forward_edges_f32(torch::Tensor rec, torch::Tensor k) {
    const at::cuda::OptionalCUDAGuard guard(device_of(rec));
    TORCH_CHECK(rec.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32);
    const int E = rec.size(0), M = k.size(0);
    auto out = torch::empty({M, 2}, rec.options().dtype(torch::kFloat64));
    edge_ft_kernel_f32<<<(M + BLOCK - 1) / BLOCK, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        rec.data_ptr<float>(), E, k.data_ptr<float>(), M, out.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("forward_double", &forward_double, "polyhedral FT, float64 face arithmetic");
    mod.def("forward_f32", &forward_f32, "polyhedral FT, all float32, compensated phase");
    mod.def("forward_edges_double", &forward_edges_double, "edge-form FT, float64");
    mod.def("forward_edges_f32", &forward_edges_f32, "edge-form FT, all float32");
    mod.def("forward_edges_float", &forward_edges_float,
            "edge-form FT, float32 with fp64 phase and near-parallel fallback");
}
