#include <iostream>
#include "sh.h"
#include "auxiliary.h"
#include "spherical_harmonics.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define CHANNELS 3

/****************************************************************************
 * 3D Spherical Harmonics forward part
 ****************************************************************************/

// This function is used in both host and device code
__host__ __device__ unsigned num_sh_bases(const unsigned degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

__device__ void sh_coeffs_to_color(
    const unsigned degree,
    const float3 &viewdir,
    const float *coeffs,
    float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // expects CHANNELS * num_bases coefficients
    // supports up to num_bases = 25
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * coeffs[9 * CHANNELS + c] +
             SH_C3[1] * xy * z * coeffs[10 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * coeffs[11 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 coeffs[12 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * coeffs[13 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * coeffs[14 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * coeffs[15 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * coeffs[16 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * coeffs[17 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * coeffs[18 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * coeffs[19 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 coeffs[20 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * coeffs[21 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 coeffs[22 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * coeffs[23 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 coeffs[24 * CHANNELS + c]);
    }
}

__global__ void compute_3dsh_fwd_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ coeffs,
    float* __restrict__ colors
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color(
        degrees_to_use, viewdirs[idx], &(coeffs[idx_sh]), &(colors[idx_col])
    );
}

__global__ void compute_3dsh_fast_fwd_kernel(const uint32_t N,
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

/****************************************************************************
 * 3D Spherical Harmonics backward part
 ****************************************************************************/

__device__ void sh_coeffs_to_color_vjp(
    const unsigned degree,
    const float3 &viewdir,
    const float *v_colors,
    float *v_coeffs
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_coeffs[1 * CHANNELS + c] = v1 * v_colors[c];
        v_coeffs[2 * CHANNELS + c] = v2 * v_colors[c];
        v_coeffs[3 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_coeffs[4 * CHANNELS + c] = v4 * v_colors[c];
        v_coeffs[5 * CHANNELS + c] = v5 * v_colors[c];
        v_coeffs[6 * CHANNELS + c] = v6 * v_colors[c];
        v_coeffs[7 * CHANNELS + c] = v7 * v_colors[c];
        v_coeffs[8 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_coeffs[9 * CHANNELS + c] = v9 * v_colors[c];
        v_coeffs[10 * CHANNELS + c] = v10 * v_colors[c];
        v_coeffs[11 * CHANNELS + c] = v11 * v_colors[c];
        v_coeffs[12 * CHANNELS + c] = v12 * v_colors[c];
        v_coeffs[13 * CHANNELS + c] = v13 * v_colors[c];
        v_coeffs[14 * CHANNELS + c] = v14 * v_colors[c];
        v_coeffs[15 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_coeffs[16 * CHANNELS + c] = v16 * v_colors[c];
        v_coeffs[17 * CHANNELS + c] = v17 * v_colors[c];
        v_coeffs[18 * CHANNELS + c] = v18 * v_colors[c];
        v_coeffs[19 * CHANNELS + c] = v19 * v_colors[c];
        v_coeffs[20 * CHANNELS + c] = v20 * v_colors[c];
        v_coeffs[21 * CHANNELS + c] = v21 * v_colors[c];
        v_coeffs[22 * CHANNELS + c] = v22 * v_colors[c];
        v_coeffs[23 * CHANNELS + c] = v23 * v_colors[c];
        v_coeffs[24 * CHANNELS + c] = v24 * v_colors[c];
    }
}

__global__ void compute_3dsh_bwd_kernel(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ v_colors,
    float* __restrict__ v_coeffs
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color_vjp(
        degrees_to_use, viewdirs[idx], &(v_colors[idx_col]), &(v_coeffs[idx_sh])
    );
}

__global__ void compute_3dsh_fast_bwd_kernel(const uint32_t N,
									  const uint32_t M, 
                                      const uint32_t D,
									  const float* __restrict__ shs,    // [N, K, 3]
									  const glm::vec3* __restrict__ dirs, 	// [N, 3]
                                      const float* __restrict__ dL_dcolor, // [N, 3]
									  float* __restrict__ dL_dsh,
									  float* __restrict__ dL_ddir
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    sh_coeffs_to_color_fast_vjp(idx, D, M, shs, dirs,
		            (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dsh, (glm::vec3*)dL_ddir);
}

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

/****************************************************************************
 * 4D Spherical Harmonics backward part
 ****************************************************************************/

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH_4D_bwd(int idx, int deg, int deg_t, int max_coeffs, const float* shs, 
			const glm::vec3* dirs, const float* dirs_t, const float time_duration,
            const glm::vec3* dL_dcolor, glm::vec3* dL_dshs, glm::vec3* dL_ddir, float* dL_ddir_t)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 dir = dirs[idx];
	const float dir_t = dirs_t[idx];

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	glm::vec3 dRGBdt(0, 0, 0);

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float l0m0 = SH_C0;

	float dRGBdsh0 = l0m0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;

	if (deg > 0){
	    float x = dir.x;
        float y = dir.y;
        float z = dir.z;

        float l1m1 = -1 * SH_C1 * y;
		float l1m0 = SH_C1 * z;
		float l1p1 = -1 * SH_C1 * x;

		float dl1m1_dy = -1 * SH_C1;
		float dl1m0_dz = SH_C1;
		float dl1p1_dx = -1 * SH_C1;

		dL_dsh[1] = l1m1 * dL_dRGB;
		dL_dsh[2] = l1m0 * dL_dRGB;
		dL_dsh[3] = l1p1 * dL_dRGB;

		dRGBdx = dl1p1_dx * sh[3];
		dRGBdy = dl1m1_dy * sh[1];
		dRGBdz = dl1m0_dz * sh[2];

		if (deg > 1){
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float l2m2 = SH_C2[0] * xy;
            float l2m1 = SH_C2[1] * yz;
            float l2m0 = SH_C2[2] * (2.0 * zz - xx - yy);
            float l2p1 = SH_C2[3] * xz;
            float l2p2 = SH_C2[4] * (xx - yy);

            float dl2m2_dx = SH_C2[0] * y;
            float dl2m2_dy = SH_C2[0] * x;
            float dl2m1_dy = SH_C2[1] * z;
            float dl2m1_dz = SH_C2[1] * y;
            float dl2m0_dx = -2 * SH_C2[2] * x;
            float dl2m0_dy = -2 * SH_C2[2] * y;
            float dl2m0_dz = 4 * SH_C2[2] * z;
            float dl2p1_dx = SH_C2[3] * z;
            float dl2p1_dz = SH_C2[3] * x;
            float dl2p2_dx = 2 * SH_C2[4] * x;
            float dl2p2_dy = -2 * SH_C2[4] * y;

			dL_dsh[4] = l2m2 * dL_dRGB;
			dL_dsh[5] = l2m1 * dL_dRGB;
			dL_dsh[6] = l2m0 * dL_dRGB;
			dL_dsh[7] = l2p1 * dL_dRGB;
			dL_dsh[8] = l2p2 * dL_dRGB;

			dRGBdx += (
				dl2m2_dx * sh[4] + dl2m0_dx * sh[6] + dl2p1_dx * sh[7] + dl2p2_dx * sh[8]
			);
			dRGBdy += (
				dl2m2_dy * sh[4] + dl2m1_dy * sh[5] + dl2m0_dy * sh[6] + dl2p2_dy * sh[8]
			);
			dRGBdz += (
				dl2m1_dz * sh[5] + dl2m0_dz * sh[6] + dl2p1_dz * sh[7]
			);

			if (deg > 2){
				float l3m3 = SH_C3[0] * y * (3 * xx - yy);
                float l3m2 = SH_C3[1] * xy * z;
                float l3m1 = SH_C3[2] * y * (4 * zz - xx - yy);
                float l3m0 = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                float l3p1 = SH_C3[4] * x * (4 * zz - xx - yy);
                float l3p2 = SH_C3[5] * z * (xx - yy);
                float l3p3 = SH_C3[6] * x * (xx - 3 * yy);

                float dl3m3_dx = SH_C3[0] * y * 6 * x;
                float dl3m3_dy = SH_C3[0] * (3 * xx - 3 * yy);
                float dl3m2_dx = SH_C3[1] * yz;
                float dl3m2_dy = SH_C3[1] * xz;
                float dl3m2_dz = SH_C3[1] * xy;
                float dl3m1_dx = -SH_C3[2] * y * 2 * x;
                float dl3m1_dy = SH_C3[2] * (4 * zz - xx - 3 * yy);
                float dl3m1_dz = SH_C3[2] * y * 8 * z;
                float dl3m0_dx = -SH_C3[3] * z * 6 * x;
                float dl3m0_dy = -SH_C3[3] * z * 6 * y;
                float dl3m0_dz = SH_C3[3] * (6 * zz - 3 * xx - 3 * yy);
                float dl3p1_dx = SH_C3[4] * (4 * zz - 3 * xx - yy);
                float dl3p1_dy = -SH_C3[4] * x * 2 * y;
                float dl3p1_dz = SH_C3[4] * x * 8 * z;
                float dl3p2_dx = SH_C3[5] * z * 2 * x;
                float dl3p2_dy = -SH_C3[5] * z * 2 * y;
                float dl3p2_dz = SH_C3[5] * (xx - yy);
                float dl3p3_dx = SH_C3[6] * (3 * xx - 3 * yy);
                float dl3p3_dy = -SH_C3[6] * x * 6 * y;

				dL_dsh[9] = l3m3 * dL_dRGB;
				dL_dsh[10] = l3m2 * dL_dRGB;
				dL_dsh[11] = l3m1 * dL_dRGB;
				dL_dsh[12] = l3m0 * dL_dRGB;
				dL_dsh[13] = l3p1 * dL_dRGB;
				dL_dsh[14] = l3p2 * dL_dRGB;
				dL_dsh[15] = l3p3 * dL_dRGB;

				dRGBdx += (
					dl3m3_dx * sh[9] +
					dl3m2_dx * sh[10] +
					dl3m1_dx * sh[11] +
					dl3m0_dx * sh[12] +
					dl3p1_dx * sh[13] +
					dl3p2_dx * sh[14] +
					dl3p3_dx * sh[15]
                );

				dRGBdy += (
                    dl3m3_dy * sh[9] +
					dl3m2_dy * sh[10] +
					dl3m1_dy * sh[11] +
					dl3m0_dy * sh[12] +
					dl3p1_dy * sh[13] +
					dl3p2_dy * sh[14] +
					dl3p3_dy * sh[15]
                );

				dRGBdz += (
					dl3m2_dz * sh[10] +
					dl3m1_dz * sh[11] +
					dl3m0_dz * sh[12] +
					dl3p1_dz * sh[13] +
					dl3p2_dz * sh[14]
                );

				if (deg_t > 0){
					float t1 = cos(2 * MY_PI * dir_t / time_duration);
					float dt1_dt = sin(2 * MY_PI * dir_t / time_duration) * 2 * MY_PI / time_duration;

					dL_dsh[16] = t1 * l0m0 * dL_dRGB;
					dL_dsh[17] = t1 * l1m1 * dL_dRGB;
					dL_dsh[18] = t1 * l1m0 * dL_dRGB;
					dL_dsh[19] = t1 * l1p1 * dL_dRGB;
					dL_dsh[20] = t1 * l2m2 * dL_dRGB;
					dL_dsh[21] = t1 * l2m1 * dL_dRGB;
					dL_dsh[22] = t1 * l2m0 * dL_dRGB;
					dL_dsh[23] = t1 * l2p1 * dL_dRGB;
					dL_dsh[24] = t1 * l2p2 * dL_dRGB;
					dL_dsh[25] = t1 * l3m3 * dL_dRGB;
					dL_dsh[26] = t1 * l3m2 * dL_dRGB;
					dL_dsh[27] = t1 * l3m1 * dL_dRGB;
					dL_dsh[28] = t1 * l3m0 * dL_dRGB;
					dL_dsh[29] = t1 * l3p1 * dL_dRGB;
					dL_dsh[30] = t1 * l3p2 * dL_dRGB;
					dL_dsh[31] = t1 * l3p3 * dL_dRGB;

					dRGBdt = dt1_dt * (
						l0m0 * sh[16] +
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

					dRGBdx += t1 * (
						dl1p1_dx * sh[19] + 
						dl2m2_dx * sh[20] + 
						dl2m0_dx * sh[22] + 
						dl2p1_dx * sh[23] + 
						dl2p2_dx * sh[24] + 
						dl3m3_dx * sh[25] +
						dl3m2_dx * sh[26] +
						dl3m1_dx * sh[27] +
						dl3m0_dx * sh[28] +
						dl3p1_dx * sh[29] +
						dl3p2_dx * sh[30] +
						dl3p3_dx * sh[31]
					);

					dRGBdy += t1 * (
						dl1m1_dy * sh[17] +
						dl2m2_dy * sh[20] + 
						dl2m1_dy * sh[21] + 
						dl2m0_dy * sh[22] + 
						dl2p2_dy * sh[24] + 
						dl3m3_dy * sh[25] +
						dl3m2_dy * sh[26] +
						dl3m1_dy * sh[27] +
						dl3m0_dy * sh[28] +
						dl3p1_dy * sh[29] +
						dl3p2_dy * sh[30] +
						dl3p3_dy * sh[31]
					);

					dRGBdz += t1 * (
						dl1m0_dz * sh[18] +
						dl2m1_dz * sh[21] + 
						dl2m0_dz * sh[22] + 
						dl2p1_dz * sh[23] +
						dl3m2_dz * sh[26] +
						dl3m1_dz * sh[27] +
						dl3m0_dz * sh[28] +
						dl3p1_dz * sh[29] +
						dl3p2_dz * sh[30]
					);

					if (deg_t > 1){
						float t2 = cos(2 * MY_PI * dir_t * 2 / time_duration);
						float dt2_dt = sin(2 * MY_PI * dir_t * 2 / time_duration) * 2 * MY_PI * 2 / time_duration;
						
						dL_dsh[32] = t2 * l0m0 * dL_dRGB;
						dL_dsh[33] = t2 * l1m1 * dL_dRGB;
						dL_dsh[34] = t2 * l1m0 * dL_dRGB;
						dL_dsh[35] = t2 * l1p1 * dL_dRGB;
						dL_dsh[36] = t2 * l2m2 * dL_dRGB;
						dL_dsh[37] = t2 * l2m1 * dL_dRGB;
						dL_dsh[38] = t2 * l2m0 * dL_dRGB;
						dL_dsh[39] = t2 * l2p1 * dL_dRGB;
						dL_dsh[40] = t2 * l2p2 * dL_dRGB;
						dL_dsh[41] = t2 * l3m3 * dL_dRGB;
						dL_dsh[42] = t2 * l3m2 * dL_dRGB;
						dL_dsh[43] = t2 * l3m1 * dL_dRGB;
						dL_dsh[44] = t2 * l3m0 * dL_dRGB;
						dL_dsh[45] = t2 * l3p1 * dL_dRGB;
						dL_dsh[46] = t2 * l3p2 * dL_dRGB;
						dL_dsh[47] = t2 * l3p3 * dL_dRGB;

						dRGBdt = dt2_dt * (
							l0m0 * sh[32] +
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

						dRGBdx += t2 * (
							dl1p1_dx * sh[35] + 
							dl2m2_dx * sh[36] + 
							dl2m0_dx * sh[38] + 
							dl2p1_dx * sh[39] + 
							dl2p2_dx * sh[40] + 
							dl3m3_dx * sh[41] +
							dl3m2_dx * sh[42] +
							dl3m1_dx * sh[43] +
							dl3m0_dx * sh[44] +
							dl3p1_dx * sh[45] +
							dl3p2_dx * sh[46] +
							dl3p3_dx * sh[47]
						);

						dRGBdy += t2 * (
							dl1m1_dy * sh[33] +
							dl2m2_dy * sh[36] + 
							dl2m1_dy * sh[37] + 
							dl2m0_dy * sh[38] + 
							dl2p2_dy * sh[40] + 
							dl3m3_dy * sh[41] +
							dl3m2_dy * sh[42] +
							dl3m1_dy * sh[43] +
							dl3m0_dy * sh[44] +
							dl3p1_dy * sh[45] +
							dl3p2_dy * sh[46] +
							dl3p3_dy * sh[47]
						);

						dRGBdz += t2 * (
							dl1m0_dz * sh[34] +
							dl2m1_dz * sh[37] + 
							dl2m0_dz * sh[38] + 
							dl2p1_dz * sh[39] +
							dl3m2_dz * sh[42] +
							dl3m1_dz * sh[43] +
							dl3m0_dz * sh[44] +
							dl3p1_dz * sh[45] +
							dl3p2_dz * sh[46]
						);
					}
				}
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	dL_ddir[idx].x = glm::dot(dRGBdx, dL_dRGB);
	dL_ddir[idx].y = glm::dot(dRGBdy, dL_dRGB);
	dL_ddir[idx].z = glm::dot(dRGBdz, dL_dRGB);

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_ddir_t[idx] = -glm::dot(dRGBdt, dL_dRGB);
}

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
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    computeColorFromSH_4D_bwd(idx, D, D_t, M, shs, dirs, dirs_t, time_duration, 
		            (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dsh, (glm::vec3*)dL_ddir, dL_ddir_t);
}
