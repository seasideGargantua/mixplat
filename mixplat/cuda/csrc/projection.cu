#include "projection.h"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3x3(glm::mat3& val, cg::thread_block_tile<32>& tile){
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            val[i][j] = cg::reduce(tile, val[i][j], cg::plus<float>());
        }
    }
}

inline __device__ void warpSum3(glm::vec3& val, cg::thread_block_tile<32>& tile){
    for (int i = 0; i < 3; i++) {
        val[i] = cg::reduce(tile, val[i], cg::plus<float>());
    }
}

/****************************************************************************
 * Projection of 3D Gaussians forward part
 ****************************************************************************/

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const unsigned block_width,
    const float clip_thresh,
    float* __restrict__ covs3d,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    float* __restrict__ compensation,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx];
    // printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y,
    // p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        // printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    float *cur_cov3d = covs3d + 6 * idx;

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d;
    float comp;
    project_cov3d_ewa(
        p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy,
        cov2d, comp
    );
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;

    // compute the projected mean
    float2 center = project_pix({fx, fy}, p_view, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max, block_width);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = center;
    compensation[idx] = comp;
    // printf(
    //     "point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d
    //     %d, tile_max %d %d\n", idx, center.x, center.y, depths[idx],
    //     radii[idx], tile_area, tile_min.x, tile_min.y, tile_max.x, tile_max.y
    // );
}

/****************************************************************************
 * Projection of 3D Gaussians backward part
 ****************************************************************************/

__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float* __restrict__ compensation,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    const float* __restrict__ v_compensation,
    float3* __restrict__ v_cov2d,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float* __restrict__ v_viewmat
) {
    auto block = cg::this_thread_block();
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = means3d[idx];
    float fx = intrins.x;
    float fy = intrins.y;
    float3 p_view = transform_4x3(viewmat, p_world);
    // get v_mean3d from v_xy
    v_mean3d[idx] = transform_4x3_rot_only_transposed(
        viewmat,
        project_pix_vjp({fx, fy}, p_view, v_xy[idx]));

    // get z gradient contribution to mean3d gradient
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    float v_z = v_depth[idx];
    v_mean3d[idx].x += viewmat[8] * v_z;
    v_mean3d[idx].y += viewmat[9] * v_z;
    v_mean3d[idx].z += viewmat[10] * v_z;

    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
    cov2d_to_compensation_vjp(compensation[idx], conics[idx], v_compensation[idx], v_cov2d[idx]);
    // get v_cov3d (and v_mean3d contribution)
    glm::mat3 v_Rot(0.f);
    glm::vec3 v_Trans(0.f);

    project_cov3d_ewa_vjp(
        p_world,
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        v_cov2d[idx],
        v_mean3d[idx],
        &(v_cov3d[6 * idx]),
        v_Rot,
        v_Trans);

    if (v_viewmat != nullptr) {
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        warpSum3x3(v_Rot, warp);
        warpSum3(v_Trans, warp);
        if (warp.thread_rank() == 0) {
            for (uint32_t i = 0; i < 3; i++) { // rows
                for (uint32_t j = 0; j < 3; j++) { // cols
                    atomicAdd(v_viewmat + i * 4 + j, v_Rot[j][i]);
                }
                atomicAdd(v_viewmat + i * 4 + 3, v_Trans[i]);
            }
        }
    }
}

/****************************************************************************
 * Projection of Gaussians utils
 ****************************************************************************/

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ void project_cov3d_ewa(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 &cov2d,
    float &compensation
) {
    // clip the
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0],viewmat[4],viewmat[8],
        viewmat[1],viewmat[5],viewmat[9],
        viewmat[2],viewmat[6],viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * std::min(lim_x, std::max(-lim_x, t.x / t.z));
    t.y = t.z * std::min(lim_y, std::max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx * rz,0.f,0.f,
        0.f,fy * rz,0.f,
        -fx * t.x * rz2,-fy * t.y * rz2,0.f
    );
    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0],cov3d[1],cov3d[2],
        cov3d[1],cov3d[3],cov3d[4],
        cov3d[2],cov3d[4],cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);

    // add a little blur along axes and save upper triangular elements
    // and compute the density compensation factor due to the blurs
    float c00 = cov[0][0], c11 = cov[1][1], c01 = cov[0][1];
    float det_orig = c00 * c11 - c01 * c01;
    cov2d.x = c00 + 0.3f;
    cov2d.y = c01;
    cov2d.z = c11 + 0.3f;
    float det_blur = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    compensation = std::sqrt(std::max(0.f, det_orig / det_blur));
}

// output space: 2D covariance, input space: cov3d
__device__ void project_cov3d_ewa_vjp(
    const float3& __restrict__ mean3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float3& __restrict__ v_cov2d,
    float3& __restrict__ v_mean3d,
    float* __restrict__ v_cov3d,
    glm::mat3& __restrict__ v_Rot,
    glm::vec3& __restrict__ v_Trans
) {
    // viewmat is row major, glm is column major
    // upper 3x3 submatrix
    // clang-format off
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    // clang-format on
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;
    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    // clang-format off
    glm::mat3 J = glm::mat3(
        fx * rz,         0.f,             0.f,
        0.f,             fy * rz,         0.f,
        -fx * t.x * rz2, -fy * t.y * rz2, 0.f
    );
    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    glm::mat3 v_cov = glm::mat3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    // clang-format on

    glm::mat3 T = J * W;
    glm::mat3 Tt = glm::transpose(T);
    glm::mat3 Vt = glm::transpose(V);
    glm::mat3 v_V = Tt * v_cov * T;
    glm::mat3 v_T = v_cov * T * Vt + glm::transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    glm::mat3 v_J = v_T * glm::transpose(W);
    float rz3 = rz2 * rz;
    glm::vec3 v_t = glm::vec3(
        -fx * rz2 * v_J[2][0],
        -fy * rz2 * v_J[2][1],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
            fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]
    );
    // printf("v_t %.2f %.2f %.2f\n", v_t[0], v_t[1], v_t[2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[0][1], W[0][2]);
    v_mean3d.x += (float)glm::dot(v_t, W[0]);
    v_mean3d.y += (float)glm::dot(v_t, W[1]);
    v_mean3d.z += (float)glm::dot(v_t, W[2]);

    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::vec3 mean3d_gl = glm::vec3(mean3d.x, mean3d.y, mean3d.z);
    glm::vec3 v_mean_c = W*glm::vec3(v_mean3d.x, v_mean3d.y, v_mean3d.z);
    v_Rot += glm::outerProduct(v_mean_c, mean3d_gl);
    v_Trans += v_mean_c;

    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    glm::mat3 v_cov_c = glm::mat3(
        v_cov2d.x,        v_cov2d.y,        0.f,
        v_cov2d.y,        v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    glm::mat3 v_V_c = glm::transpose(J) * v_cov * J;
    v_Rot += v_V_c * W * glm::transpose(V) + glm::transpose(v_V_c) * W * V;
}