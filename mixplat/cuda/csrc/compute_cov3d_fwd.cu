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

torch::Tensor computeCov3D_fwd_tensor(
    torch::Tensor scales, 
    const float glob_scale, 
    torch::Tensor quats
) {
    const int num_points = scales.size(0);
    torch::Tensor Cov3Ds = torch::zeros({num_points, 6}, scales.options().dtype(torch::kFloat32));
    computeCov3D_fwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3*)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4*)quats.contiguous().data_ptr<float>(),
        Cov3Ds.contiguous().data_ptr<float>()
    );

    return Cov3Ds;
}

}