#include <torch/extension.h>
#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spherical_harmonics_3d_forward", &mixplat::compute_3dsh_forward_tensor);
  m.def("spherical_harmonics_3d_backward", &mixplat::compute_3dsh_backward_tensor);
  m.def("spherical_harmonics_4d_forward", &mixplat::compute_4dsh_forward_tensor);
  m.def("spherical_harmonics_4d_backward", &mixplat::compute_4dsh_backward_tensor);
  m.def("compute_3d_covariance_forward", &mixplat::computeCov3D_fwd_tensor);
  m.def("compute_3d_covariance_backward", &mixplat::computeCov3D_bwd_tensor);
  m.def("compute_4d_covariance_forward", &mixplat::computeCov3D_conditional_fwd_tensor);
  m.def("compute_4d_covariance_backward", &mixplat::computeCov3D_conditional_bwd_tensor);
  m.def("project_gaussians_forward", &mixplat::project_gaussians_forward_tensor);
  m.def("project_gaussians_backward", &mixplat::project_gaussians_backward_tensor);
  m.def("rasterize_forward", &mixplat::rasterize_forward_tensor);
  m.def("rasterize_backward", &mixplat::rasterize_backward_tensor);
  m.def("rasterize_to_indices_in_range", &mixplat::rasterize_to_indices_in_range_tensor);
  m.def("rasterize_to_pixels_fwd", &mixplat::rasterize_to_pixels_fwd_tensor);
  m.def("rasterize_to_pixels_bwd", &mixplat::rasterize_to_pixels_bwd_tensor);
  // utils
  m.def("map_gaussian_to_intersects", &mixplat::map_gaussian_to_intersects_tensor);
  m.def("get_tile_bin_edges", &mixplat::get_tile_bin_edges_tensor);
  m.def("compute_relocation", &mixplat::compute_relocation_tensor);
  m.def("isect_tiles", &mixplat::isect_tiles_tensor);
  m.def("isect_offset_encode", &mixplat::isect_offset_encode_tensor);
}