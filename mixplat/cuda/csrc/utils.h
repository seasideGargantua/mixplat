#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "types.cuh"
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"

/****************************************************************************
 * Scale and rotation to covariance matrix 3D
 ****************************************************************************/

__global__ void computeCov3D_fwd_kernel(
    const int num_points,
    const float3* scales, 
    const float glob_scale, 
    const float4* quats, 
    float *cov3ds
);

__global__ void computeCov3D_bwd_kernel(
    const int num_points,
    const float3* scales,
    const float glob_scale,
    const float4* quats,
    const float* __restrict__ v_cov3ds,
    float3* __restrict__ v_scales,
    float4* __restrict__ v_quats
);

/****************************************************************************
 * Scale and rotation to covariance matrix 4D
 ****************************************************************************/

__global__ void computeCov3D_conditional_fwd_kernel(
    const int num_points,
    const glm::vec3* scales, 
    const float* scale_ts, 
    const float glob_scale,
	const glm::vec4* rots, 
    const glm::vec4* rot_rs, 
    float* cov3Ds,
    float* cov_ts, 
    glm::vec3* speed);

__global__ void computeCov3D_conditional_bwd_kernel(
    const int num_points, 
    const glm::vec3* scales, 
    const float* scale_ts, 
    float glob_scale,
    const glm::vec4* rots, 
    const glm::vec4* rot_rs, 
    const float* dL_dcov3Ds, 
    const glm::vec3* dL_dspeed, 
    const float* dL_dcov_t,
    glm::vec3* dL_dscales, 
    float* dL_dscales_t, 
    glm::vec4* dL_drots, 
    glm::vec4* dL_drots_r);

/****************************************************************************
 * Compute relocation in 3DGS MCMC
 ****************************************************************************/

__global__ void compute_relocation_kernel(
    int P, 
    float* opacity_old, 
    float* scale_old,
    float* scale_t_old,
    int* N, 
    float* binoms, 
    int n_max, 
    float* opacity_new, 
    float* scale_new,
    float* scale_t_new);

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

template <typename T>
__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
);

__global__ void isect_offset_encode(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
);