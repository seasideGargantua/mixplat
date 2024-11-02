#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>

#define N_THREADS 256

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

/****************************************************************************
 * 3D Spherical Harmonics
 ****************************************************************************/

torch::Tensor compute_3dsh_forward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
);

torch::Tensor compute_3dsh_fast_forward_tensor(
    const unsigned num_points,
    const unsigned D,
    const torch::Tensor &shs,
    const torch::Tensor &dirs
);

torch::Tensor compute_3dsh_backward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
);

std::tuple<
    torch::Tensor,
    torch::Tensor>
compute_3dsh_fast_backward_tensor(
    const unsigned num_points,
    const unsigned D,
    const torch::Tensor &shs,
    const torch::Tensor &dirs,
    torch::Tensor &dL_dcolor
);

/****************************************************************************
 * 4D Spherical Harmonics
 ****************************************************************************/

torch::Tensor compute_4dsh_forward_tensor(
    const unsigned num_points,
    const unsigned D,
    const unsigned D_t,
    const torch::Tensor &shs,
    const torch::Tensor &dirs,
    const torch::Tensor &dirs_t,
    const float time_duration
);

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
);

/****************************************************************************
 * Compute 3D Gaussians Covariance
 ****************************************************************************/

torch::Tensor computeCov3D_fwd_tensor(
    torch::Tensor scales, 
    const float glob_scale, 
    torch::Tensor quats
);

std::tuple<torch::Tensor, torch::Tensor> computeCov3D_bwd_tensor(
    torch::Tensor scales, 
    const float glob_scale, 
    torch::Tensor quats,
    torch::Tensor v_cov3ds
);

/****************************************************************************
 * Compute relocation in 3DGS MCMC
 ****************************************************************************/

std::tuple<torch::Tensor, 
          torch::Tensor,
          torch::Tensor> 
compute_relocation_tensor(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
    torch::Tensor& scale_t_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max);

/****************************************************************************
 * Projection of 3D Gaussians
 ****************************************************************************/

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &cov3d,
    torch::Tensor &viewmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width,
    const float clip_thresh
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &viewmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &compensation,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic,
    torch::Tensor &v_compensation
);

/****************************************************************************
 * Compute 4D Gaussians Covariance
 ****************************************************************************/

std::tuple<torch::Tensor, 
          torch::Tensor,
          torch::Tensor>
computeCov3D_conditional_fwd_tensor(
    torch::Tensor scales,
    torch::Tensor scale_ts,
    const float glob_scale, 
    torch::Tensor quats,
    torch::Tensor quats_r
);

std::tuple<torch::Tensor, 
          torch::Tensor,
          torch::Tensor,
          torch::Tensor> 
computeCov3D_conditional_bwd_tensor(
    torch::Tensor scales,
    torch::Tensor scale_ts,
    const float glob_scale, 
    torch::Tensor quats,
    torch::Tensor quats_r,
    torch::Tensor dL_dcov3Ds,
    torch::Tensor dL_dcov_ts,
    torch::Tensor dL_dspeed
);

/****************************************************************************
 * Rasteruzation of Gaussians
 ****************************************************************************/

std::tuple<torch::Tensor, torch::Tensor> map_gaussian_to_intersects_tensor(
    const int num_points,
    const int num_intersects,
    const torch::Tensor &xys,
    const torch::Tensor &depths,
    const torch::Tensor &radii,
    const torch::Tensor &cum_tiles_hit,
    const std::tuple<int, int, int> tile_bounds,
    const unsigned block_width
);

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted, 
    const std::tuple<int, int, int> tile_bounds
);

std::tuple<
        torch::Tensor, 
        torch::Tensor, 
        torch::Tensor,
        torch::Tensor>
rasterize_forward_tensor(
    const std::tuple<int, int, int> tile_bounds,
    const std::tuple<int, int, int> block,
    const std::tuple<int, int, int> img_size,
    const bool return_invdepth,
    const torch::Tensor &gaussian_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &depths,
    const torch::Tensor &opacities,
    const torch::Tensor &background,
    const torch::Tensor &interp_weights,
    const torch::Tensor &kid_nodes
);

std::tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor, // dL_dopacity
        torch::Tensor  // dL_dinvdepth
        >
rasterize_backward_tensor(
    const unsigned img_height,
    const unsigned img_width,
    const unsigned block_width,
    const bool return_invdepth,
    const torch::Tensor &interp_weights,
    const torch::Tensor &kid_nodes,
    const torch::Tensor &gaussians_ids_sorted,
    const torch::Tensor &tile_bins,
    const torch::Tensor &xys,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &depths,
    const torch::Tensor &opacities,
    const torch::Tensor &background,
    const torch::Tensor &final_Ts,
    const torch::Tensor &final_idx,
    const torch::Tensor &v_output, // dL_dout_color
    const torch::Tensor &v_output_alpha, // dL_dout_alpha
    const torch::Tensor &v_output_invdepth // dL_dout_invdepth
);
