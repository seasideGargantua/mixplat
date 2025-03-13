#ifndef MIXPLAT_CUDA_HELPERS_CUH
#define MIXPLAT_CUDA_HELPERS_CUH

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <cuda_runtime.h>
#include <iostream>

namespace mixplat {

namespace cg = cooperative_groups;

inline __device__ void get_bbox(const float2 center, const float2 dims,
                                const dim3 img_size, uint2 &bb_min, uint2 &bb_max) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline __device__ void get_tile_bbox(const float2 pix_center, const float pix_radius,
                                     const dim3 tile_bounds, uint2 &tile_min,
                                     uint2 &tile_max, const int block_size) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {pix_center.x / (float)block_size,
                          pix_center.y / (float)block_size};
    float2 tile_radius = {pix_radius / (float)block_size,
                          pix_radius / (float)block_size};
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

inline __device__ bool compute_cov2d_bounds(const float3 cov2d, float3 &conic,
                                            float &radius) {
    // find eigenvalues of 2d covariance matrix
    // expects upper triangular values of cov matrix as float3
    // then compute the radius and conic dimensions
    // the conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f)
        return false;
    float inv_det = 1.f / det;

    // inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// compute vjp from df/d_conic to df/c_cov2d
inline __device__ void cov2d_to_conic_vjp(const float3 &conic, const float3 &v_conic,
                                          float3 &v_cov2d) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    glm::mat2 X = glm::mat2(conic.x, conic.y, conic.y, conic.z);
    glm::mat2 G = glm::mat2(v_conic.x, v_conic.y / 2.f, v_conic.y / 2.f, v_conic.z);
    glm::mat2 v_Sigma = -X * G * X;
    v_cov2d.x = v_Sigma[0][0];
    v_cov2d.y = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d.z = v_Sigma[1][1];
}

inline __device__ void cov2d_to_compensation_vjp(const float compensation,
                                                 const float3 &conic,
                                                 const float v_compensation,
                                                 float3 &v_cov2d) {
    // comp = sqrt(det(cov2d - 0.3 I) / det(cov2d))
    // conic = inverse(cov2d)
    // df / d_cov2d = df / d comp * 0.5 / comp * [ d comp^2 / d cov2d ]
    // d comp^2 / d cov2d = (1 - comp^2) * conic - 0.3 I * det(conic)
    float inv_det = conic.x * conic.z - conic.y * conic.y;
    float one_minus_sqr_comp = 1 - compensation * compensation;
    float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    v_cov2d.x += v_sqr_comp * (one_minus_sqr_comp * conic.x - 0.3 * inv_det);
    v_cov2d.y += 2 * v_sqr_comp * (one_minus_sqr_comp * conic.y);
    v_cov2d.z += v_sqr_comp * (one_minus_sqr_comp * conic.z - 0.3 * inv_det);
}

// helper for applying R^T * p for a ROW MAJOR 4x3 matrix [R, t], ignoring t
inline __device__ float3 transform_4x3_rot_only_transposed(const float *mat,
                                                           const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[4] * p.y + mat[8] * p.z,
        mat[1] * p.x + mat[5] * p.y + mat[9] * p.z,
        mat[2] * p.x + mat[6] * p.y + mat[10] * p.z,
    };
    return out;
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline __device__ float3 transform_4x3(const float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline __device__ float4 transform_4x4(const float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline __device__ float2 project_pix(const float2 fxfy, const float3 p_view,
                                     const float2 pp) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 p_proj = {p_view.x * rw, p_view.y * rw};
    float2 p_pix = {p_proj.x * fxfy.x + pp.x, p_proj.y * fxfy.y + pp.y};
    return p_pix;
}

// given v_xy_pix, get v_xyz
inline __device__ float3 project_pix_vjp(const float2 fxfy, const float3 p_view,
                                         const float2 v_xy) {
    float rw = 1.f / (p_view.z + 1e-6f);
    float2 v_proj = {fxfy.x * v_xy.x, fxfy.y * v_xy.y};
    float3 v_view = {v_proj.x * rw, v_proj.y * rw,
                     -(v_proj.x * p_view.x + v_proj.y * p_view.y) * rw * rw};
    return v_view;
}

inline __device__ glm::mat3 quat_to_rotmat(const float4 quat) {
    // quat to rotation matrix
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;

    // glm matrices are column-major
    return glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y));
}

inline __device__ float4 quat_to_rotmat_vjp(const float4 quat, const glm::mat3 v_R) {
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;

    float4 v_quat;
    // v_R is COLUMN MAJOR
    // w element stored in x field
    v_quat.x = 2.f * (
                         // v_quat.w = 2.f * (
                         x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                         z * (v_R[0][1] - v_R[1][0]));
    // x element in y field
    v_quat.y =
        2.f * (
                  // v_quat.x = 2.f * (
                  -2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
                  z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1]));
    // y element in z field
    v_quat.z =
        2.f * (
                  // v_quat.y = 2.f * (
                  x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
                  z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2]));
    // z element in w field
    v_quat.w =
        2.f * (
                  // v_quat.z = 2.f * (
                  x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
                  2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]));
    return v_quat;
}

inline __device__ glm::mat3 scale_to_mat(const float3 scale, const float glob_scale) {
    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// device helper for culling near points
inline __device__ bool clip_near_plane(const float3 p, const float *viewmat,
                                       float3 &p_view, float thresh) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

template <typename T>
inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R,
    const vec3<T> t,
    const vec3<T> p,
    vec3<T> &p_c
) {
    p_c = R * p + t;
}

template <typename T>
inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R,
    const vec3<T> t,
    const vec3<T> p,
    // grad outputs
    const vec3<T> v_p_c,
    // grad inputs
    mat3<T> &v_R,
    vec3<T> &v_t,
    vec3<T> &v_p
) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

template <typename T>
inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R,
    const mat3<T> covar,
    mat3<T> &covar_c
) {
    covar_c = R * covar * glm::transpose(R);
}

template <typename T>
inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R,
    const mat3<T> covar,
    // grad outputs
    const mat3<T> v_covar_c,
    // grad inputs
    mat3<T> &v_R,
    mat3<T> &v_covar
) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R += v_covar_c * R * glm::transpose(covar) +
           glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

template <typename T>
inline __device__ T inverse(const mat2<T> M, mat2<T> &Minv) {
    T det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    T invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

template <typename T>
inline __device__ void inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

template <typename T>
inline __device__ T add_blur(const T eps2d, mat2<T> &covar, T &compensation) {
    T det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    T det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

template <typename T>
inline __device__ void add_blur_vjp(
    const T eps2d,
    const mat2<T> conic_blur,
    const T compensation,
    const T v_compensation,
    mat2<T> &v_covar
) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    T det_conic_blur = conic_blur[0][0] * conic_blur[1][1] -
                       conic_blur[0][1] * conic_blur[1][0];
    T v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    T one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] -
                                   eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] -
                                   eps2d * det_conic_blur);
}

template <typename T>
inline __device__ void persp_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});
}

template <typename T>
inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3<T>(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] +
                  2.f * fy * ty * rz3 * v_J[2][1];
}

template <uint32_t DIM, class T, class WarpT>
inline __device__ void warpSum(T *val, WarpT &warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<T>());
    }
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec4<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
    val.z = cg::reduce(warp, val.z, cg::plus<ScalarT>());
    val.w = cg::reduce(warp, val.w, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec3<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
    val.z = cg::reduce(warp, val.z, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec2<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat4<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat3<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat2<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpMax(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<ScalarT>());
}

template <typename T> __forceinline__ __device__ T sum(vec3<T> a) {
    return a.x + a.y + a.z;
}


}

#endif // MIXPLAT_CUDA_HELPERS_CUH