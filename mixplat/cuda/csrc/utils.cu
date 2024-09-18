// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#include "utils.h"
#include "helpers.cuh"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>
namespace cg = cooperative_groups;

/****************************************************************************
 * Scale and rotation to covariance matrix 3D
 ****************************************************************************/

// device helper to get 3D covariance from scale and quat parameters
__global__ void computeCov3D_fwd_kernel(
    const int num_points,
    const float3* scales, 
    const float glob_scale, 
    const float4* quats, 
    float *cov3ds
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    float* cov3d = cov3ds + idx * 6;
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__global__ void computeCov3D_bwd_kernel(
    const int num_points,
    const float3* scales,
    const float glob_scale,
    const float4* quats,
    const float* __restrict__ v_cov3ds,
    float3* __restrict__ v_scales,
    float4* __restrict__ v_quats
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    
    const float3 scale = scales[idx];
    const float4 quat = quats[idx];
    const float* v_cov3d = v_cov3ds + idx * 6;
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scales[idx].x = (float)glm::dot(R[0], v_M[0]) * glob_scale;
    v_scales[idx].y = (float)glm::dot(R[1], v_M[1]) * glob_scale;
    v_scales[idx].z = (float)glm::dot(R[2], v_M[2]) * glob_scale;

    glm::mat3 v_R = v_M * S;
    v_quats[idx] = quat_to_rotmat_vjp(quat, v_R);
}

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

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
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
    glm::vec4* dL_drots_r)
{
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    const glm::vec3 scale = scales[idx];
    const float scale_t = scale_ts[idx];
    const glm::vec4 rot = rots[idx];
    const glm::vec4 rot_r = rot_rs[idx];

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

    glm::mat3 cov11 = glm::mat3(Sigma);
    glm::vec3 cov12 = glm::vec3(Sigma[0][3],Sigma[1][3],Sigma[2][3]);

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dL_dcov12 = -glm::vec3(
		dL_dcov3D[0] * cov12[0] + dL_dcov3D[1] * cov12[1]*0.5 + dL_dcov3D[2] * cov12[2]*0.5,
		dL_dcov3D[1] * cov12[0]*0.5 + dL_dcov3D[3] * cov12[1] + dL_dcov3D[4] * cov12[2]*0.5,
		dL_dcov3D[2] * cov12[0]*0.5 + dL_dcov3D[4] * cov12[1]*0.5 + dL_dcov3D[5] * cov12[2]
	) * 2.0f / cov_t;

	dL_dcov12 += dL_dspeed[idx] / cov_t;

    float dL_dcov_t_w_ms_cov = dL_dcov_t[idx];
	float dL_dms_dot_cov12 = glm::dot(dL_dspeed[idx], cov12);
	dL_dcov_t_w_ms_cov += -dL_dms_dot_cov12 / (cov_t * cov_t);
	dL_dcov_t_w_ms_cov += (
		cov12[0] * cov12[0] * dL_dcov3D[0] + cov12[0] * cov12[1] * dL_dcov3D[1] +
		cov12[0] * cov12[2] * dL_dcov3D[2] + cov12[1] * cov12[1] * dL_dcov3D[3] +
		cov12[1] * cov12[2] * dL_dcov3D[4] + cov12[2] * cov12[2] * dL_dcov3D[5]
    ) / (cov_t * cov_t);

    glm::mat4 dL_dSigma = glm::mat4(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2], 0.5f * dL_dcov12[0],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4], 0.5f * dL_dcov12[1],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5], 0.5f * dL_dcov12[2],
		0.5f * dL_dcov12[0], 0.5f * dL_dcov12[1], 0.5f * dL_dcov12[2], dL_dcov_t_w_ms_cov
	);
	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat4 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat4 Rt = glm::transpose(R);
	glm::mat4 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
	dL_dscales_t[idx] = glm::dot(Rt[3], dL_dMt[3]);

	dL_dMt[0] *= glob_scale * scale.x;
	dL_dMt[1] *= glob_scale * scale.y;
	dL_dMt[2] *= glob_scale * scale.z;
	dL_dMt[3] *= glob_scale * scale_t;

	glm::mat4 dL_dml_t = dL_dMt * M_r;
    // dL_dml_t = glm::transpose(dL_dml_t);
    glm::vec4 dL_drot(0.0f);
	dL_drot.x = dL_dml_t[0][0] + dL_dml_t[1][1] + dL_dml_t[2][2] + dL_dml_t[3][3];
	dL_drot.y = dL_dml_t[0][1] - dL_dml_t[1][0] + dL_dml_t[2][3] - dL_dml_t[3][2];
	dL_drot.z = dL_dml_t[0][2] - dL_dml_t[1][3] - dL_dml_t[2][0] + dL_dml_t[3][1];
	dL_drot.w = dL_dml_t[0][3] + dL_dml_t[1][2] - dL_dml_t[2][1] - dL_dml_t[3][0];

    glm::mat4 dL_dmr_t = M_l * dL_dMt;
    glm::vec4 dL_drot_r(0.0f);
	dL_drot_r.x = dL_dmr_t[0][0] + dL_dmr_t[1][1] + dL_dmr_t[2][2] + dL_dmr_t[3][3];
	dL_drot_r.y = -dL_dmr_t[0][1] + dL_dmr_t[1][0] + dL_dmr_t[2][3] - dL_dmr_t[3][2];
	dL_drot_r.z = -dL_dmr_t[0][2] - dL_dmr_t[1][3] + dL_dmr_t[2][0] + dL_dmr_t[3][1];
	dL_drot_r.w = -dL_dmr_t[0][3] + dL_dmr_t[1][2] - dL_dmr_t[2][1] + dL_dmr_t[3][0];
    
    float4 dL_drot_f = dnormvdv(float4{rot.x, rot.y, rot.z, rot.w}, float4{dL_drot.x, dL_drot.y, dL_drot.z, dL_drot.w});
	float4 dL_drot_r_f = dnormvdv(float4{rot_r.x, rot_r.y, rot_r.z, rot_r.w}, float4{dL_drot_r.x, dL_drot_r.y, dL_drot_r.z, dL_drot_r.w});

    dL_drots[idx].x = dL_drot_f.x;
    dL_drots[idx].y = dL_drot_f.y;
    dL_drots[idx].z = dL_drot_f.z;
    dL_drots[idx].w = dL_drot_f.w;

    dL_drots_r[idx].x = dL_drot_r_f.x;
    dL_drots_r[idx].y = dL_drot_r_f.y;
    dL_drots_r[idx].z = dL_drot_r_f.z;
    dL_drots_r[idx].w = dL_drot_r_f.w;
}

