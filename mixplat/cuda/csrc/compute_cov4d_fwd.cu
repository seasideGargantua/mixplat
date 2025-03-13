#include "bindings.h"
#include "helpers.cuh"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace mixplat {

namespace cg = cooperative_groups;

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
    glm::vec3* speed)
{
	unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    const glm::vec3 scale = scales[idx];
    const float scale_t = scale_ts[idx];
    const glm::vec4 rot = rots[idx];
    const glm::vec4 rot_r = rot_rs[idx];
    float* cov3D = cov3Ds + idx * 6;

    // Create scaling matrix
	glm::mat4 S = glm::mat4(1.0f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	S[2][2] = glob_scale * scale.z;
	S[3][3] = glob_scale * scale_t;

	const float l_l = glm::length(rot);
	const float a = rot.x / l_l;
	const float b = rot.y / l_l;
	const float c = rot.z / l_l;
	const float d = rot.w / l_l;

    const float l_r = glm::length(rot_r);
	const float p = rot_r.x / l_r;
	const float q = rot_r.y / l_r;
	const float r = rot_r.z / l_r;
	const float s = rot_r.w / l_r;

	glm::mat4 M_l = glm::mat4(
		a, -b, -c, -d,
		b, a,-d, c,
		c, d, a,-b,
		d,-c, b, a
	);

	glm::mat4 M_r = glm::mat4(
		p, q, r, s,
		-q, p,-s, r,
		-r, s, p,-q,
		-s,-r, q, p
	);

	// glm stores in column major
	glm::mat4 R = M_r * M_l;
	glm::mat4 M = S * R;
	glm::mat4 Sigma = glm::transpose(M) * M;
	float cov_t = Sigma[3][3];
    cov_ts[idx] = cov_t;

	glm::mat3 cov11 = glm::mat3(Sigma);
	glm::vec3 cov12 = glm::vec3(Sigma[0][3],Sigma[1][3],Sigma[2][3]);
	glm::mat3 cov3D_condition = cov11 - glm::outerProduct(cov12, cov12) / cov_t;

	// Covariance is symmetric, only store upper right
	cov3D[0] = cov3D_condition[0][0];
	cov3D[1] = cov3D_condition[0][1];
	cov3D[2] = cov3D_condition[0][2];
	cov3D[3] = cov3D_condition[1][1];
	cov3D[4] = cov3D_condition[1][2];
	cov3D[5] = cov3D_condition[2][2];
    speed[idx] = cov12 / cov_t;
}

std::tuple<torch::Tensor, 
          torch::Tensor,
          torch::Tensor>
computeCov3D_conditional_fwd_tensor(
    torch::Tensor scales,
    torch::Tensor scale_ts,
    const float glob_scale, 
    torch::Tensor quats,
    torch::Tensor quats_r
) {
    const int num_points = scales.size(0);
    torch::Tensor speed = torch::zeros({num_points, 3}, scales.options().dtype(torch::kFloat32));
    torch::Tensor Cov3Ds = torch::zeros({num_points, 6}, scales.options().dtype(torch::kFloat32));
    torch::Tensor Cov_ts = torch::zeros({num_points, 1}, scales.options().dtype(torch::kFloat32));
    computeCov3D_conditional_fwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (glm::vec3*)scales.contiguous().data_ptr<float>(),
        scale_ts.contiguous().data_ptr<float>(),
        glob_scale,
        (glm::vec4*)quats.contiguous().data_ptr<float>(),
        (glm::vec4*)quats_r.contiguous().data_ptr<float>(),
        Cov3Ds.contiguous().data_ptr<float>(),
        Cov_ts.contiguous().data_ptr<float>(),
        (glm::vec3*)speed.contiguous().data_ptr<float>()
    );

    return std::make_tuple(Cov3Ds, Cov_ts, speed);
}

}