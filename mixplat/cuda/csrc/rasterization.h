#include <cuda.h>
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include "types.cuh"
#include <cstdio>
#include <iostream>
#define MAX_BLOCK_SIZE ( 16 * 16 )

/****************************************************************************
 * Rasterization of Gaussians utils
 ****************************************************************************/

__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
);

__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t* __restrict__ isect_ids_sorted, int2* __restrict__ tile_bins
);

/****************************************************************************
 * Rasterization of Gaussians forward part
 ****************************************************************************/

__global__ void rasterize_forward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const bool return_invdepth,
    const float* interp_ts,
	const int* kids,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ colors,
    const float* __restrict__ depths,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    float* __restrict__ out_invdepth,
    const float3& __restrict__ background
);

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
);

/****************************************************************************
 * Rasterization to Indices in Range
 ****************************************************************************/

template <typename T>
__global__ void rasterize_to_indices_in_range_kernel(
    const uint32_t range_start,
    const uint32_t range_end,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2<T> *__restrict__ means2d, // [C, N, 2]
    const vec3<T> *__restrict__ conics,  // [C, N, 3]
    const T *__restrict__ opacities,     // [C, N]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const T *__restrict__ transmittances,     // [C, image_height, image_width]
    const int32_t *__restrict__ chunk_starts, // [C, image_height, image_width]
    int32_t *__restrict__ chunk_cnts,         // [C, image_height, image_width]
    int64_t *__restrict__ gaussian_ids,       // [n_elems]
    int64_t *__restrict__ pixel_ids           // [n_elems]
);