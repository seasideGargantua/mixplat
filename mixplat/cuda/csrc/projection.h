#include <cuda.h>
#include <cuda_runtime.h>
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include "types.cuh"
#include <cstdio>
#include <iostream>

/****************************************************************************
 * Projection of 3D Gaussians
 ****************************************************************************/

template <typename T>
__global__ void project_gaussians_forward_kernel(
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 6] optional
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    const T near_plane,
    const T far_plane,
    const T radius_clip,
    // outputs
    int32_t *__restrict__ radii,  // [C, N]
    T *__restrict__ means2d,      // [C, N, 2]
    T *__restrict__ depths,       // [C, N]
    T *__restrict__ conics,       // [C, N, 3]
    T *__restrict__ compensations // [C, N] optional
);

template <typename T>
__global__ void project_gaussians_backward_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,    // [N, 3]
    const T *__restrict__ covars,   // [N, 6]
    const T *__restrict__ viewmats, // [C, 4, 4]
    const T *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const T eps2d,
    // fwd outputs
    const int32_t *__restrict__ radii,   // [C, N]
    const T *__restrict__ conics,        // [C, N, 3]
    const T *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const T *__restrict__ v_means2d,       // [C, N, 2]
    const T *__restrict__ v_depths,        // [C, N]
    const T *__restrict__ v_conics,        // [C, N, 3]
    const T *__restrict__ v_compensations, // [C, N] optional
    // grad inputs
    T *__restrict__ v_means,   // [N, 3]
    T *__restrict__ v_covars,  // [N, 6] optional
    T *__restrict__ v_viewmats // [C, 4, 4] optional
);