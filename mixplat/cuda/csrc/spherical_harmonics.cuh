#include "bindings.h"
#include "types.cuh"

#include <cuda_runtime.h>


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

__device__ void sh_coeffs_to_color_fast_vjp(int idx, int deg, int max_coeffs, const float* shs, 
			const glm::vec3* dirs, const glm::vec3* dL_dcolor, glm::vec3* dL_dshs, glm::vec3* dL_ddir)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 dir = dirs[idx];

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
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	dL_ddir[idx].x = glm::dot(dRGBdx, dL_dRGB);
	dL_ddir[idx].y = glm::dot(dRGBdy, dL_dRGB);
	dL_ddir[idx].z = glm::dot(dRGBdz, dL_dRGB);

}
