from typing import Callable


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


rasterize_forward = _make_lazy_cuda_func("rasterize_forward")
rasterize_backward = _make_lazy_cuda_func("rasterize_backward")
compute_3d_covariance_forward = _make_lazy_cuda_func("compute_3d_covariance_forward")
compute_3d_covariance_backward = _make_lazy_cuda_func("compute_3d_covariance_backward")
compute_4d_covariance_forward = _make_lazy_cuda_func("compute_4d_covariance_forward")
compute_4d_covariance_backward = _make_lazy_cuda_func("compute_4d_covariance_backward")
project_gaussians_forward = _make_lazy_cuda_func("project_gaussians_forward")
project_gaussians_backward = _make_lazy_cuda_func("project_gaussians_backward")
spherical_harmonics_3d_forward = _make_lazy_cuda_func("spherical_harmonics_3d_forward")
spherical_harmonics_3d_backward = _make_lazy_cuda_func("spherical_harmonics_3d_backward")
spherical_harmonics_4d_forward = _make_lazy_cuda_func("spherical_harmonics_4d_forward")
spherical_harmonics_4d_backward = _make_lazy_cuda_func("spherical_harmonics_4d_backward")
map_gaussian_to_intersects = _make_lazy_cuda_func("map_gaussian_to_intersects")
get_tile_bin_edges = _make_lazy_cuda_func("get_tile_bin_edges")
reorder_data_forward = _make_lazy_cuda_func("reorder_data_forward")
reorder_data_backward = _make_lazy_cuda_func("reorder_data_backward")
compute_relocation = _make_lazy_cuda_func("compute_relocation")
