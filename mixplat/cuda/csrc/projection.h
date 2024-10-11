#include <cuda.h>
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cstdio>
#include <iostream>

/****************************************************************************
 * Projection of 3D Gaussians
 ****************************************************************************/

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
);

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
);

/****************************************************************************
 * Projection of 3D Gaussians utils
 ****************************************************************************/

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
);

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
);