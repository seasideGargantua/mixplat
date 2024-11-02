#include <torch/extension.h>
#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spherical_harmonics_3d_forward", &compute_3dsh_forward_tensor);
  m.def("spherical_harmonics_3d_backward", &compute_3dsh_backward_tensor);
  m.def("spherical_harmonics_3d_fast_forward", &compute_3dsh_fast_forward_tensor);
  m.def("spherical_harmonics_3d_fast_backward", &compute_3dsh_fast_backward_tensor);
  m.def("spherical_harmonics_4d_forward", &compute_4dsh_forward_tensor);
  m.def("spherical_harmonics_4d_backward", &compute_4dsh_backward_tensor);
  m.def("compute_3d_covariance_forward", &computeCov3D_fwd_tensor);
  m.def("compute_3d_covariance_backward", &computeCov3D_bwd_tensor);
  m.def("compute_4d_covariance_forward", &computeCov3D_conditional_fwd_tensor);
  m.def("compute_4d_covariance_backward", &computeCov3D_conditional_bwd_tensor);
  m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
  m.def("project_gaussians_backward", &project_gaussians_backward_tensor);
  m.def("rasterize_forward", &rasterize_forward_tensor);
  m.def("rasterize_backward", &rasterize_backward_tensor);
  // utils
  m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
  m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);
  m.def("compute_relocation", &compute_relocation_tensor);
}