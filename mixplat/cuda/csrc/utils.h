#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
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