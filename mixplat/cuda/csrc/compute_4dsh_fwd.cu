#include "bindings.h"
#include "auxiliary.h"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mixplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * 4D Spherical Harmonics forward part
 ****************************************************************************/

// Forward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ glm::vec3 computeColorFromSH_4D_fwd(int idx, int deg, int deg_t, int max_coeffs, const float* shs, const glm::vec3* dirs, const float* dirs_t, const float time_duration)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 dir = dirs[idx];
	const float dir_t = dirs_t[idx];

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

				if (deg_t > 0){
					float t1 = cos(2 * MY_PI * dir_t / time_duration);

					result += t1 * (l0m0 * sh[16] +
						l1m1 * sh[17] +
						l1m0 * sh[18] +
						l1p1 * sh[19] + 
						l2m2 * sh[20] +
						l2m1 * sh[21] +
						l2m0 * sh[22] +
						l2p1 * sh[23] +
						l2p2 * sh[24] + 
						l3m3 * sh[25] +
						l3m2 * sh[26] +
						l3m1 * sh[27] +
						l3m0 * sh[28] +
						l3p1 * sh[29] +
						l3p2 * sh[30] +
						l3p3 * sh[31]);

					if (deg_t > 1){
						float t2 = cos(2 * MY_PI * dir_t * 2 / time_duration);

						result += t2 * (l0m0 * sh[32] +
							l1m1 * sh[33] +
							l1m0 * sh[34] +
							l1p1 * sh[35] + 
							l2m2 * sh[36] +
							l2m1 * sh[37] +
							l2m0 * sh[38] +
							l2p1 * sh[39] +
							l2p2 * sh[40] + 
							l3m3 * sh[41] +
							l3m2 * sh[42] +
							l3m1 * sh[43] +
							l3m0 * sh[44] +
							l3p1 * sh[45] +
							l3p2 * sh[46] +
							l3p3 * sh[47]);
					}

				}
			}
		}
	}
	result += 0.5f;

	return result;
}

__global__ void compute_4dsh_fwd_kernel(const uint32_t N,
									  const uint32_t M,
                                      const uint32_t D,
                                      const uint32_t D_t,
									  const glm::vec3* dir, 
									  const float* dir_t,
                                      const float* __restrict__ shs, // [N, M, 3]
                                      float* __restrict__ colors,        // [N, 3]
								  	  const float time_duration
) { 
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    glm::vec3 result = computeColorFromSH_4D_fwd(idx, D, D_t, M, shs, dir, dir_t, time_duration);
    colors[idx * 3 + 0] = result.x;
    colors[idx * 3 + 1] = result.y;
    colors[idx * 3 + 2] = result.z;
}

torch::Tensor compute_4dsh_forward_tensor(
    const unsigned num_points,
    const unsigned D,
    const unsigned D_t,
    const torch::Tensor &shs,
    const torch::Tensor &dirs,
    const torch::Tensor &dirs_t,
    const float time_duration
) {
    int M = 0;
    if(shs.size(0) != 0)
    {	
        M = shs.size(1);
    }
    torch::Tensor colors = 
        torch::zeros({num_points, 3}, shs.options().dtype(torch::kFloat32));  
    compute_4dsh_fwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        M,
        D,
        D_t,
        (glm::vec3*)dirs.contiguous().data_ptr<float>(),
        dirs_t.contiguous().data_ptr<float>(),
        shs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        time_duration
    );
    return colors;
}

}