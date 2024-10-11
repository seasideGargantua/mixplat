#include "bindings.h"
#include "sh.h"
#include "projection.h"
#include "rasterization.h"
#include "helpers.cuh"
#include "utils.h"
#include <cstdio>
#include <iostream>
#include <torch/extension.h>
#include <tuple>

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
) {
    DEVICE_GUARD(means3d);
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = int((img_width + block_width - 1) / block_width);
    tile_bounds_dim3.y = int((img_height + block_width - 1) / block_width);
    tile_bounds_dim3.z = 1;

    float4 intrins = {fx, fy, cx, cy};

    // Triangular covariance.
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor compensation_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        tile_bounds_dim3,
        block_width,
        clip_thresh,
        cov3d.contiguous().data_ptr<float>(),
        // Outputs.
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        compensation_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        xys_d, depths_d, radii_d, conics_d, compensation_d, num_tiles_hit_d
    );
}

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
){
    DEVICE_GUARD(means3d);
    dim3 img_size_dim3;
    img_size_dim3.x = img_width;
    img_size_dim3.y = img_height;

    float4 intrins = {fx, fy, cx, cy};

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_viewmat =
        torch::zeros({3, 4}, means3d.options().dtype(torch::kFloat32));

    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        intrins,
        img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float *)compensation.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        v_depth.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        (float *)v_compensation.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        v_cov3d.contiguous().data_ptr<float>(),
        (float3 *)v_mean3d.contiguous().data_ptr<float>(),
        v_viewmat.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_viewmat);
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

