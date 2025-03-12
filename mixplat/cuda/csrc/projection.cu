#include "projection.h"
#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

/****************************************************************************
 * Projection of 3D Gaussians forward part
 ****************************************************************************/

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
template <typename T>
__global__ void project_gaussians_forward_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 6] optional
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    // outputs
    int32_t *__restrict__ radii,  // [C, N]
    T *__restrict__ means2d,      // [C, N, 2]
    T *__restrict__ depths,       // [C, N]
    T *__restrict__ conics,       // [C, N, 3]
    T *__restrict__ compensations // [C, N] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3<T> R = mat3<T>(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3<T> covar;
    covars += gid * 6;
    covar = mat3<T>(
        covars[0],
        covars[1],
        covars[2], // 1st column
        covars[1],
        covars[3],
        covars[4], // 2nd column
        covars[2],
        covars[4],
        covars[5] // 3rd column
    );
    
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // perspective projection
    mat2<T> covar2d;
    vec2<T> mean2d;

    persp_proj<T>(
        mean_c,
        covar_c,
        Ks[0],
        Ks[4],
        Ks[2],
        Ks[5],
        image_width,
        image_height,
        covar2d,
        mean2d
    );
            
    T compensation;
    T det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2<T> covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // take 3 sigma as the radius (non differentiable)
    T b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    T v1 = b + sqrt(max(0.01f, b * b - det));
    T radius = ceil(3.f * sqrt(v1));
    // T v2 = b - sqrt(max(0.1f, b * b - det));
    // T radius = ceil(3.f * sqrt(max(v1, v2)));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

/****************************************************************************
 * Projection of 3D Gaussians backward part
 ****************************************************************************/

template <typename T>
__global__ void project_gaussians_backward_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 6]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    // fwd outputs
    const int32_t *__restrict__ radii,   // [C, N]
    const T *__restrict__ conics,        // [C, N, 3]
    const T *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const T *__restrict__ v_means2d,       // [C, N, 2]
    const T *__restrict__ v_depths,        // [C, N]
    const T *__restrict__ v_conics,        // [C, N, 3]
    const T *__restrict__ v_compensations, // [C, N] optional
    // grad inputs
    T *__restrict__ v_means,   // [N, 3]
    T *__restrict__ v_covars,  // [N, 6] optional
    T *__restrict__ v_viewmats // [C, 4, 4] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2<T> covar2d_inv = mat2<T>(conics[0], conics[1], conics[1], conics[2]);
    mat2<T> v_covar2d_inv =
        mat2<T>(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2<T> v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const T compensation = compensations[idx];
        const T v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3<T> R = mat3<T>(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    mat3<T> covar;
    covars += gid * 6;
    covar = mat3<T>(
        covars[0],
        covars[1],
        covars[2], // 1st column
        covars[1],
        covars[3],
        covars[4], // 2nd column
        covars[2],
        covars[4],
        covars[5] // 3rd column
    );
    vec3<T> mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    mat3<T> covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // vjp: perspective projection
    T fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3<T> v_covar_c(0.f);
    vec3<T> v_mean_c(0.f);

    persp_proj_vjp<T>(
        mean_c,
        covar_c,
        fx,
        fy,
        cx,
        cy,
        image_width,
        image_height,
        v_covar2d,
        glm::make_vec2(v_means2d),
        v_mean_c,
        v_covar_c
    );

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3<T> v_mean(0.f);
    mat3<T> v_covar(0.f);
    mat3<T> v_R(0.f);
    vec3<T> v_t(0.f);
    pos_world_to_cam_vjp(
        R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
    );
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }

    // Output gradients w.r.t. the covariance matrix
    warpSum(v_covar, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_covars += gid * 6;
        gpuAtomicAdd(v_covars, v_covar[0][0]);
        gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
        gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
        gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
        gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
        gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
    }
    
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}