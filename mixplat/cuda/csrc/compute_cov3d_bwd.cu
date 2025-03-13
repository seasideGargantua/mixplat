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
 * Scale and rotation to covariance matrix 3D
 ****************************************************************************/

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

std::tuple<torch::Tensor, torch::Tensor> computeCov3D_bwd_tensor(
    torch::Tensor scales, 
    const float glob_scale, 
    torch::Tensor quats,
    torch::Tensor v_cov3ds
) {
    const int num_points = scales.size(0);
    torch::Tensor v_scales = torch::zeros({num_points, 3}, scales.options().dtype(torch::kFloat32));
    torch::Tensor v_quats = torch::zeros({num_points, 4}, scales.options().dtype(torch::kFloat32));
    computeCov3D_bwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3*)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4*)quats.contiguous().data_ptr<float>(),
        v_cov3ds.contiguous().data_ptr<float>(),
        (float3*)v_scales.contiguous().data_ptr<float>(),
        (float4*)v_quats.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_scales, v_quats);
}

}