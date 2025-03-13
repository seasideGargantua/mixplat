#include "bindings.h"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace mixplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Compute relocation in 3DGS MCMC
 ****************************************************************************/

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ void compute_relocation_kernel(
    int P, 
    float* opacity_old, 
    float* scale_old,
    float* scale_t_old,
    int* N, 
    float* binoms, 
    int n_max, 
    float* opacity_new, 
    float* scale_new,
    float* scale_t_new) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P) return;
    
    int N_idx = N[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    opacity_new[idx] = 1.0f - powf(1.0f - opacity_old[idx], 1.0f / N_idx);
    
    // compute new scale
    for (int i = 1; i <= N_idx; ++i) {
        for (int k = 0; k <= (i-1); ++k) {
            float bin_coeff = binoms[(i-1) * n_max + k];
            float term = (pow(-1, k) / sqrt(k + 1)) * pow(opacity_new[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (opacity_old[idx] / denom_sum);
    for (int i = 0; i < 3; ++i)
        scale_new[idx * 3 + i] = coeff * scale_old[idx * 3 + i];
    
    if (scale_t_old != nullptr) {
        scale_t_new[idx] = coeff * scale_t_old[idx];
    }
}

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

}