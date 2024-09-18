#include "cuda_runtime.h"
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cstdio>
#include <iostream>

/****************************************************************************
 * 3D Spherical Harmonics
 ****************************************************************************/

__host__ __device__ unsigned num_sh_bases(const unsigned degree);

__global__ void compute_3dsh_fwd_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ coeffs,
    float* __restrict__ colors
);

__global__ void compute_3dsh_bwd_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ v_colors,
    float* __restrict__ v_coeffs
);

/****************************************************************************
 * 4D Spherical Harmonics
 ****************************************************************************/

__global__ void compute_4dsh_fwd_kernel(const uint32_t N,
									  const uint32_t M,
                                      const uint32_t D,
                                      const uint32_t D_t,
									  const glm::vec3* dir, 
									  const float* dir_t,
                                      const float* __restrict__ shs, // [N, M, 3]
                                      float* __restrict__ colors,        // [N, 3]
								  	  const float time_duration
);

__global__ void compute_4dsh_bwd_kernel(const uint32_t N,
									  const uint32_t M, 
                                      const uint32_t D,
                                      const uint32_t D_t,
									  const float* __restrict__ shs,    // [N, K, 3]
									  const glm::vec3* __restrict__ dirs, 	// [N, 3]
									  const float* __restrict__ dirs_t, // [N, 1]
									  const float time_duration,
                                      const float* __restrict__ dL_dcolor, // [N, 3]
									  float* __restrict__ dL_dsh,
									  float* __restrict__ dL_ddir,
									  float* __restrict__ dL_ddir_t
);

