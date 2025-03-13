#include "bindings.h"
#include "auxiliary.h"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mixplat {

namespace cg = cooperative_groups;

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

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
compute_4dsh_backward_tensor(
    const unsigned num_points,
    const unsigned D,
    const unsigned D_t,
    const torch::Tensor &shs,
    const torch::Tensor &dirs,
    const torch::Tensor &dirs_t,
    const float time_duration,
    torch::Tensor &dL_dcolor
) {
    int M = 0;
    if(shs.size(0) != 0)
    {	
        M = shs.size(1);
    }
    torch::Tensor dL_dsh =
        torch::zeros({num_points, M, 3}, shs.options().dtype(torch::kFloat32));
    torch::Tensor dL_ddir = 
        torch::zeros({num_points, 3}, shs.options().dtype(torch::kFloat32));
    torch::Tensor dL_ddir_t = 
        torch::zeros({num_points, 1}, shs.options().dtype(torch::kFloat32));

    
    compute_4dsh_bwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        M,
        D,
        D_t,
        shs.contiguous().data_ptr<float>(),
        (glm::vec3*)dirs.contiguous().data_ptr<float>(),
        dirs_t.contiguous().data_ptr<float>(),
        time_duration,
        dL_dcolor.contiguous().data_ptr<float>(),
        dL_dsh.contiguous().data_ptr<float>(),
        dL_ddir.contiguous().data_ptr<float>(),
        dL_ddir_t.contiguous().data_ptr<float>()
    );
    return std::make_tuple(dL_dsh, dL_ddir, dL_ddir_t);
}

}