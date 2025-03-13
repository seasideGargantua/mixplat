#include "bindings.h"
#include "auxiliary.h"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace mixplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * 3D Spherical Harmonics forward part
 ****************************************************************************/

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation
__device__ glm::vec3 sh_coeffs_to_color_fast(int idx, int deg, int max_coeffs, const float* shs, const glm::vec3* dirs)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir = dirs[idx];

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	float l0m0=SH_C0;
	glm::vec3 result = l0m0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		float l1m1 = -1 * SH_C1 * y;
		float l1m0 = SH_C1 * z;
		float l1p1 = -1 * SH_C1 * x;

		result += 
			l1m1 * sh[1] +
			l1m0 * sh[2] +
			l1p1 * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float l2m2 = SH_C2[0] * xy;
            float l2m1 = SH_C2[1] * yz;
            float l2m0 = SH_C2[2] * (2.0 * zz - xx - yy);
            float l2p1 = SH_C2[3] * xz;
            float l2p2 = SH_C2[4] * (xx - yy);

			result +=
                l2m2 * sh[4] +
                l2m1 * sh[5] +
                l2m0 * sh[6] +
                l2p1 * sh[7] +
                l2p2 * sh[8];

			if (deg > 2)
			{
				float l3m3 = SH_C3[0] * y * (3 * xx - yy);
                float l3m2 = SH_C3[1] * xy * z;
                float l3m1 = SH_C3[2] * y * (4 * zz - xx - yy);
                float l3m0 = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                float l3p1 = SH_C3[4] * x * (4 * zz - xx - yy);
                float l3p2 = SH_C3[5] * z * (xx - yy);
                float l3p3 = SH_C3[6] * x * (xx - 3 * yy);

				result +=
					l3m3 * sh[9] +
					l3m2 * sh[10] +
					l3m1 * sh[11] +
					l3m0 * sh[12] +
					l3p1 * sh[13] +
					l3p2 * sh[14] +
					l3p3 * sh[15];
			}
		}
	}
	result += 0.5f;

	return result;
}

__global__ void compute_3dsh_fwd_kernel(const uint32_t N,
									  const uint32_t M,
                                      const uint32_t D,
									  const glm::vec3* dir, 
                                      const float* __restrict__ shs, // [N, M, 3]
                                      float* __restrict__ colors        // [N, 3]
) { 
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    glm::vec3 result = sh_coeffs_to_color_fast(idx, D, M, shs, dir);
    colors[idx * 3 + 0] = result.x;
    colors[idx * 3 + 1] = result.y;
    colors[idx * 3 + 2] = result.z;
}

torch::Tensor compute_3dsh_forward_tensor(
    const unsigned num_points,
    const unsigned D,
    const torch::Tensor &shs,
    const torch::Tensor &dirs
) {
    int M = 0;
    if(shs.size(0) != 0)
    {	
        M = shs.size(1);
    }
    torch::Tensor colors = 
        torch::zeros({num_points, 3}, shs.options().dtype(torch::kFloat32));  
    compute_3dsh_fwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        M,
        D,
        (glm::vec3*)dirs.contiguous().data_ptr<float>(),
        shs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

}