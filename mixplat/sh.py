import torch
import mixplat.cuda as _C

#------------------------------------------------------------#
# Define the C++/CUDA 3d spherical harmonics class and API   #
#------------------------------------------------------------#

def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25


def deg_from_sh(num_bases: int):
    if num_bases == 1:
        return 0
    if num_bases == 4:
        return 1
    if num_bases == 9:
        return 2
    if num_bases == 16:
        return 3
    if num_bases == 25:
        return 4
    assert False, "Invalid number of SH bases"


def spherical_harmonics_3d(
    degrees_to_use: int,
    viewdirs,
    coeffs,
):
    """Compute spherical harmonics

    Note:
        This function is only differentiable to the input coeffs.

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.

    Returns:
        The spherical harmonics.
    """
    assert coeffs.shape[-2] >= num_sh_bases(degrees_to_use)
    return _3DSphericalHarmonics.apply(
        degrees_to_use, viewdirs.contiguous(), coeffs.contiguous()
    )


class _3DSphericalHarmonics(torch.autograd.Function):
    """Compute spherical harmonics

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.
    """

    @staticmethod
    def forward(
        ctx,
        degrees_to_use: int,
        viewdirs,
        coeffs,
    ):
        num_points = coeffs.shape[0]
        ctx.degrees_to_use = degrees_to_use
        degree = deg_from_sh(coeffs.shape[-2])
        ctx.degree = degree
        ctx.save_for_backward(viewdirs)
        return _C.spherical_harmonics_3d_forward(
            num_points, degree, degrees_to_use, viewdirs, coeffs
        )

    @staticmethod
    def backward(ctx, v_colors):
        degrees_to_use = ctx.degrees_to_use
        degree = ctx.degree
        viewdirs = ctx.saved_tensors[0]
        num_points = v_colors.shape[0]
        return (
            None,
            None,
            _C.spherical_harmonics_3d_backward(
                num_points, degree, degrees_to_use, viewdirs, v_colors
            ),
        )

def spherical_harmonics_3d_fast(
    degrees_to_use: int,
    viewdirs,
    coeffs,
):
    """Compute spherical harmonics

    Note:
        This function is only differentiable to the input coeffs.

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.

    Returns:
        The spherical harmonics.
    """
    assert coeffs.shape[-2] >= num_sh_bases(degrees_to_use)
    return _3DSphericalHarmonicsFast.apply(
        degrees_to_use, coeffs.contiguous(), viewdirs.contiguous()
    )


class _3DSphericalHarmonicsFast(torch.autograd.Function):
    """Compute spherical harmonics

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.
    """

    @staticmethod
    def forward(
        ctx, degree, shs, dirs
    ):
        num_points = shs.shape[0]
        color = _C.spherical_harmonics_3d_fast_forward(
            num_points, 
            degree, 
            shs, 
            dirs
        )
        ctx.save_for_backward(shs, dirs)
        ctx.num_points = num_points
        ctx.degree = degree
        return color

    @staticmethod
    def backward(ctx, dL_dcolors):
        shs, dirs = ctx.saved_tensors
        num_points = ctx.num_points
        degree = ctx.degree
        dL_dsh, dL_ddir = _C.spherical_harmonics_3d_fast_backward(
                num_points,
                degree,
                shs,
                dirs,
                dL_dcolors
            )
        return (
            None,
            dL_dsh, 
            dL_ddir,
        )

#------------------------------------------------------------#
# Define the C++/CUDA 4d spherical harmonics class and API   #
#------------------------------------------------------------#

def spherical_harmonics_4d(
    degree,
    degree_t,
    shs,  # [..., K, 3]
    dirs,   # [..., 3]
    dirs_t, # [..., 1]
    time_duration
):
    """Computes spherical harmonics.

    Args:
        degree: The degree to be used.
        degree_t: The degree of time dimention to be used.
        shs: Spherical harmonics coefficients. [..., K, 3]
        dirs: View directions. [..., 3]
        dirs_t: Time directions. [..., 1]
        time_duration: The duration of the current frame.

    Returns:
        colors. [..., 3]
    """
    assert (degree + 1) ** 2 <= shs.shape[-2], shs.shape
    assert shs.shape[-1] == 3, shs.shape
    return _4DSphericalHarmonics.apply(
        degree, degree_t, shs, dirs, dirs_t, time_duration
    )

class _4DSphericalHarmonics(torch.autograd.Function):
    """4D Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, degree, degree_t, shs, dirs, dirs_t, time_duration
    ):
        num_points = shs.shape[0]
        colors = _C.spherical_harmonics_4d_forward(
            num_points, 
            degree, 
            degree_t, 
            shs, 
            dirs, 
            dirs_t, 
            time_duration)
        ctx.save_for_backward(shs, dirs, dirs_t)
        ctx.num_points = num_points
        ctx.degree = degree
        ctx.degree_t = degree_t
        ctx.time_duration = time_duration
        return colors

    @staticmethod
    def backward(ctx, dL_dcolors):
        shs, dirs, dirs_t = ctx.saved_tensors
        num_points = ctx.num_points
        degree = ctx.degree
        degree_t = ctx.degree_t
        time_duration = ctx.time_duration
        dL_dsh, dL_ddir, dL_ddir_t = _C.spherical_harmonics_4d_backward(
            num_points,
            degree,
            degree_t,
            shs,
            dirs,
            dirs_t,
            time_duration,
            dL_dcolors
        )
        return None, None, dL_dsh, dL_ddir, dL_ddir_t, None