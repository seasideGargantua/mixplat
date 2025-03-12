#include "bindings.h"
#include "sh.h"
#include "projection.h"
#include "rasterization.h"
#include "helpers.cuh"
#include "utils.h"
#include "types.cuh"
#include <cstdio>
#include <iostream>
#include <torch/extension.h>
#include <tuple>
#include <cub/cub.cuh>

/****************************************************************************
 * 3D Spherical Harmonics
 ****************************************************************************/

torch::Tensor compute_3dsh_forward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    DEVICE_GUARD(viewdirs);
    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3 || coeffs.size(0) != num_points ||
        coeffs.size(1) != num_bases || coeffs.size(2) != 3) {
        AT_ERROR("coeffs must have dimensions (N, D, 3)");
    }
    torch::Tensor colors = torch::empty({num_points, 3}, coeffs.options());
    compute_3dsh_fwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        coeffs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

torch::Tensor compute_3dsh_backward_tensor(
    const unsigned num_points,
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    DEVICE_GUARD(viewdirs);
    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points ||
        viewdirs.size(1) != 3) {
        AT_ERROR("viewdirs must have dimensions (N, 3)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points ||
        v_colors.size(1) != 3) {
        AT_ERROR("v_colors must have dimensions (N, 3)");
    }
    unsigned num_bases = num_sh_bases(degree);
    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, 3}, v_colors.options());
    compute_3dsh_bwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_coeffs.contiguous().data_ptr<float>()
    );
    return v_coeffs;
}

torch::Tensor compute_3dsh_fast_forward_tensor(
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
    compute_3dsh_fast_fwd_kernel<<<
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

std::tuple<
    torch::Tensor,
    torch::Tensor>
compute_3dsh_fast_backward_tensor(
    const unsigned num_points,
    const unsigned D,
    const torch::Tensor &shs,
    const torch::Tensor &dirs,
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

    
    compute_3dsh_fast_bwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        M,
        D,
        shs.contiguous().data_ptr<float>(),
        (glm::vec3*)dirs.contiguous().data_ptr<float>(),
        dL_dcolor.contiguous().data_ptr<float>(),
        dL_dsh.contiguous().data_ptr<float>(),
        dL_ddir.contiguous().data_ptr<float>()
    );
    return std::make_tuple(dL_dsh, dL_ddir);
}

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

/****************************************************************************
 * Scale and rotation to covariance matrix 3D
 ****************************************************************************/

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

/****************************************************************************
 * Scale and rotation to covariance matrix 4D
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
) {
    const int num_points = scales.size(0);
    torch::Tensor dL_dscales = 
        torch::zeros({num_points, 3}, scales.options().dtype(torch::kFloat32));
    torch::Tensor dL_dscales_t = 
        torch::zeros({num_points, 1}, scales.options().dtype(torch::kFloat32));
    torch::Tensor dL_drotations = 
        torch::zeros({num_points, 4}, scales.options().dtype(torch::kFloat32));
    torch::Tensor dL_drotations_r = 
        torch::zeros({num_points, 4}, scales.options().dtype(torch::kFloat32));
    computeCov3D_conditional_bwd_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (glm::vec3*)scales.contiguous().data_ptr<float>(),
        scale_ts.contiguous().data_ptr<float>(),
        glob_scale,
        (glm::vec4*)quats.contiguous().data_ptr<float>(),
        (glm::vec4*)quats_r.contiguous().data_ptr<float>(),
        dL_dcov3Ds.contiguous().data_ptr<float>(),
        (glm::vec3*)dL_dspeed.contiguous().data_ptr<float>(),
        dL_dcov_ts.contiguous().data_ptr<float>(),
        (glm::vec3*)dL_dscales.contiguous().data_ptr<float>(),
        dL_dscales_t.contiguous().data_ptr<float>(),
        (glm::vec4*)dL_drotations.contiguous().data_ptr<float>(),
        (glm::vec4*)dL_drotations_r.contiguous().data_ptr<float>()
    );

    return std::make_tuple(dL_dscales, dL_dscales_t, dL_drotations, dL_drotations_r);
}

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
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));
    torch::Tensor final_scale_t = torch::full({P}, 0, scale_t_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		compute_relocation_kernel<<<
        (P + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
            P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
            scale_t_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>(),
            final_scale_t.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale, final_scale_t);

}

/****************************************************************************
 * Projection of 3D Gaussians
 ****************************************************************************/

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars.value());
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    if (N) {
        project_gaussians_forward_kernel<float>
            <<<(N + N_THREADS - 1) / N_THREADS,
               N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.value().data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>(),
                calc_compensations ? compensations.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,                       // [C, N]
    const torch::Tensor &conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &compensations, // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars.value());
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars = torch::zeros_like(covars.value()); // optional
    
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (N) {
        project_gaussians_backward_kernel<float>
            <<<(N + N_THREADS - 1) / N_THREADS,
               N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.value().data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                radii.data_ptr<int32_t>(),
                conics.data_ptr<float>(),
                compensations.has_value()
                    ? compensations.value().data_ptr<float>()
                    : nullptr,
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_compensations.has_value()
                    ? v_compensations.value().data_ptr<float>()
                    : nullptr,
                v_means.data_ptr<float>(),
                v_covars.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_covars, v_viewmats);
}

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
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(depths);
    CHECK_INPUT(radii);
    CHECK_INPUT(cum_tiles_hit);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    torch::Tensor gaussian_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt32));
    torch::Tensor isect_ids_unsorted =
        torch::zeros({num_intersects}, xys.options().dtype(torch::kInt64));

    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float2 *)xys.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        cum_tiles_hit.contiguous().data_ptr<int32_t>(),
        tile_bounds_dim3,
        block_width,
        // Outputs.
        isect_ids_unsorted.contiguous().data_ptr<int64_t>(),
        gaussian_ids_unsorted.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(isect_ids_unsorted, gaussian_ids_unsorted);
}

torch::Tensor get_tile_bin_edges_tensor(
    int num_intersects, const torch::Tensor &isect_ids_sorted, 
    const std::tuple<int, int, int> tile_bounds
) {
    DEVICE_GUARD(isect_ids_sorted);
    CHECK_INPUT(isect_ids_sorted);
    int num_tiles = std::get<0>(tile_bounds) * std::get<1>(tile_bounds);
    torch::Tensor tile_bins = torch::zeros(
        {num_tiles, 2}, isect_ids_sorted.options().dtype(torch::kInt32)
    );
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_intersects,
        isect_ids_sorted.contiguous().data_ptr<int64_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>()
    );
    return tile_bins;
}

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
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(gaussian_ids_sorted);
    CHECK_INPUT(tile_bins);
    CHECK_INPUT(xys);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(background);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    dim3 block_dim3;
    block_dim3.x = std::get<0>(block);
    block_dim3.y = std::get<1>(block);
    block_dim3.z = std::get<2>(block);

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);
    img_size_dim3.z = std::get<2>(img_size);

    const int channels = colors.size(1);
    const int img_width = img_size_dim3.x;
    const int img_height = img_size_dim3.y;

    torch::Tensor out_img = torch::zeros(
        {img_height, img_width, channels}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor out_invdepth = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_Ts = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kFloat32)
    );
    torch::Tensor final_idx = torch::zeros(
        {img_height, img_width}, xys.options().dtype(torch::kInt32)
    );

    rasterize_forward_kernel<<<tile_bounds_dim3, block_dim3>>>(
        tile_bounds_dim3,
        img_size_dim3,
        return_invdepth,
        interp_weights.contiguous().data_ptr<float>(),
        kid_nodes.contiguous().data_ptr<int>(),
        gaussian_ids_sorted.contiguous().data_ptr<int32_t>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)out_img.contiguous().data_ptr<float>(),
        out_invdepth.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>()
    );

    return std::make_tuple(out_img, out_invdepth, final_Ts, final_idx);
}

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
) {
    DEVICE_GUARD(xys);
    CHECK_INPUT(xys);
    CHECK_INPUT(colors);

    if (xys.ndimension() != 2 || xys.size(1) != 2) {
        AT_ERROR("xys must have dimensions (num_points, 2)");
    }

    if (colors.ndimension() != 2 || colors.size(1) != 3) {
        AT_ERROR("colors must have 2 dimensions");
    }

    const int num_points = xys.size(0);
    const dim3 tile_bounds = {
        (img_width + block_width - 1) / block_width,
        (img_height + block_width - 1) / block_width,
        1
    };
    const dim3 block(block_width, block_width, 1);
    const dim3 img_size = {img_width, img_height, 1};
    const int channels = colors.size(1);

    torch::Tensor v_xy = torch::zeros({num_points, 2}, xys.options());
    torch::Tensor v_conic = torch::zeros({num_points, 3}, xys.options());
    torch::Tensor v_colors =
        torch::zeros({num_points, channels}, xys.options());
    torch::Tensor v_opacity = torch::zeros({num_points, 1}, xys.options());
    torch::Tensor v_depth = torch::zeros({num_points}, xys.options());

    rasterize_backward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        return_invdepth,
        interp_weights.contiguous().data_ptr<float>(),
        kid_nodes.contiguous().data_ptr<int>(),
        gaussians_ids_sorted.contiguous().data_ptr<int>(),
        (int2 *)tile_bins.contiguous().data_ptr<int>(),
        (float2 *)xys.contiguous().data_ptr<float>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float3 *)colors.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        opacities.contiguous().data_ptr<float>(),
        *(float3 *)background.contiguous().data_ptr<float>(),
        final_Ts.contiguous().data_ptr<float>(),
        final_idx.contiguous().data_ptr<int>(),
        (float3 *)v_output.contiguous().data_ptr<float>(),
        v_output_alpha.contiguous().data_ptr<float>(),
        v_output_invdepth.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float3 *)v_colors.contiguous().data_ptr<float>(),
        v_opacity.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_xy, v_conic, v_colors, v_opacity, v_depth);
}

/****************************************************************************
 * Rasterization to Indices in Range
 ****************************************************************************/

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
    const uint32_t range_start,
    const uint32_t range_end,           // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &opacities, // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    uint32_t C = means2d.size(0); // number of cameras
    uint32_t N = means2d.size(1); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));
    if (cudaFuncSetAttribute(
            rasterize_to_indices_in_range_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    torch::Tensor chunk_starts;
    if (n_isects) {
        torch::Tensor chunk_cnts = torch::zeros(
            {C * image_height * image_width},
            means2d.options().dtype(torch::kInt32)
        );
        rasterize_to_indices_in_range_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                range_start,
                range_end,
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                opacities.data_ptr<float>(),
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                transmittances.data_ptr<float>(),
                nullptr,
                chunk_cnts.data_ptr<int32_t>(),
                nullptr,
                nullptr
            );

        torch::Tensor cumsum =
            torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = cumsum - chunk_cnts;
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    torch::Tensor gaussian_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    torch::Tensor pixel_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    if (n_elems) {
        rasterize_to_indices_in_range_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                range_start,
                range_end,
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                opacities.data_ptr<float>(),
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                transmittances.data_ptr<float>(),
                chunk_starts.data_ptr<int32_t>(),
                nullptr,
                gaussian_ids.data_ptr<int64_t>(),
                pixel_ids.data_ptr<int64_t>()
            );
    }
    return std::make_tuple(gaussian_ids, pixel_ids);
}

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                      // [C, N] or [nnz]
    const torch::Tensor &depths,                     // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_total_elems",
            [&]() {
                isect_tiles<<<
                    (total_elems + N_THREADS - 1) / N_THREADS,
                    N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.data_ptr<int32_t>(),
                    nullptr,
                    nullptr
                );
            }
        );
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_n_isects",
            [&]() {
                isect_tiles<<<
                    (total_elems + N_THREADS - 1) / N_THREADS,
                    N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.data_ptr<int64_t>(),
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    nullptr,
                    isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>()
                );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor &isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    uint32_t n_isects = isect_ids.size(0);
    torch::Tensor offsets = torch::empty(
        {C, tile_height, tile_width}, isect_ids.options().dtype(torch::kInt32)
    );
    if (n_isects) {
        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        isect_offset_encode<<<
            (n_isects + N_THREADS - 1) / N_THREADS,
            N_THREADS,
            0,
            stream>>>(
            n_isects,
            isect_ids.data_ptr<int64_t>(),
            C,
            n_tiles,
            tile_n_bits,
            offsets.data_ptr<int32_t>()
        );
    } else {
        offsets.fill_(0);
    }
    return offsets;
}