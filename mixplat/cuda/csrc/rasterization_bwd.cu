#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include "auxiliary.h"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>

namespace mixplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization of Gaussians utils
 ****************************************************************************/

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

/****************************************************************************
 * Rasterization of Gaussians backward part
 ****************************************************************************/


__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const bool return_invdepth,
    const float* __restrict__ interp_ts,
	const int* __restrict__ kids,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ depths,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    const float* __restrict__ v_output_invdepth,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity,
    float* __restrict__ v_depth
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = inside ? final_Ts[pix_id] : 0;
    float T = T_final;
    // the contribution from gaussians behind the current one
    // float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside ? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float depths_batch[MAX_BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_out_alpha = v_output_alpha[pix_id];
    float v_out_invdepth;
    float3 accum_rec = { 0.f, 0.f, 0.f };
	float accum_invdepth_rec = 0.f;
    if (return_invdepth){
        v_out_invdepth = v_output_invdepth[pix_id];
    }
    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    float last_alpha = 0.f;
	float3 last_color = { 0.f, 0.f, 0.f };
	float last_invdepth = 0.f;

    bool do_interp = (interp_ts != nullptr && kids != nullptr); 

    // Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * img_size.x;
	const float ddely_dy = 0.5 * img_size.y;

    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            if(return_invdepth) depths_batch[tr] = depths[g_id];
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            bool nullalpha;
            float opac;
            float2 delta;
            float3 conic;
            float vis;
            if(valid){
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                float tmp_alpha = opac * vis;
                nullalpha = tmp_alpha > 0.999f;
                float raw_alpha = min(0.999f, tmp_alpha);
                if (do_interp)
                {
                    int global_id = id_batch[t];
                    float interp = interp_ts[global_id];

                    float frac =  1.0f / kids[global_id];
                    float kidsqrt_alpha = 1.0f - pow(1.0f - raw_alpha, frac);
                    alpha = interp * raw_alpha + (1.0f - interp) * kidsqrt_alpha;
                }
                else
                {
                    alpha = raw_alpha;
                }

                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid)){
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_invdepth_local = 0.f;
            float v_depth_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid){
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};
                

                const float3 rgb = rgbs_batch[t];

                accum_rec.x = last_alpha * last_color.x + (1.f - last_alpha) * accum_rec.x;
                accum_rec.y = last_alpha * last_color.y + (1.f - last_alpha) * accum_rec.y;
                accum_rec.z = last_alpha * last_color.z + (1.f - last_alpha) * accum_rec.z;

				last_color.x = rgb.x;
                last_color.y = rgb.y;
                last_color.z = rgb.z;

                // contribution from this pixel
                v_alpha += (rgb.x - accum_rec.x) * v_out.x;
                v_alpha += (rgb.y - accum_rec.y) * v_out.y;
                v_alpha += (rgb.z - accum_rec.z) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                

                if (return_invdepth)
                {
                const float invd = 1.f / depths_batch[t];
                accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
                last_invdepth = invd;
                v_alpha += (invd - accum_invdepth_rec) * v_out_invdepth;
                v_invdepth_local = fac * v_out_invdepth;
                v_depth_local = -v_invdepth_local*invd*invd;
                }
                
                v_alpha *= T;
                last_alpha = alpha;

                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;

                // update the running sum
                // buffer.x += rgb.x * fac;
                // buffer.y += rgb.y * fac;
                // buffer.z += rgb.z * fac;
                v_alpha = nullalpha ? 0 : v_alpha;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x, 
                                 0.5f * v_sigma * delta.x * delta.y,
                                 0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {  v_sigma * (conic.x * delta.x + conic.y * delta.y)*ddelx_dx, 
                                v_sigma * (conic.y * delta.x + conic.z * delta.y)*ddely_dy};
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if (return_invdepth) warpSum(v_depth_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);
                
                atomicAdd(v_opacity + g, v_opacity_local);

                // Propagate gradients from inverse depth to alphaas and
                // per Gaussian inverse depths
                if (return_invdepth)
                {
                float* v_depth_ptr = (float*)(v_depth);
                atomicAdd(v_depth_ptr + g, v_depth_local);
                }

            }
        }
    }
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

}