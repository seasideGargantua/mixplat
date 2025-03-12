import torch
import mixplat.cuda as _C

#---------------------------------------------------------------------#
# Define the C++/CUDA Compute 3D gaussians covariance class and API   #
#---------------------------------------------------------------------#

def compute_3d_gaussians_covariance(
    scales,
    quats,
    glob_scale = 1.,
):
    """This function computes the 3D gaussians covariance.

    Note:
        This function is differentiable w.r.t the scales and quats inputs.

    Args:
       means_grad (Tensor): xyzs grad of gaussians.
       cov3d (Tensor): 3D covariances of gaussians.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor}:

        - **cov4d** (Tensor): 4D covariances.    
        - **mean4d** (Tensor): 4D means.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"
    return _Compute3DCovariance.apply(
        scales,
        quats,
        glob_scale
    )

class _Compute3DCovariance(torch.autograd.Function):
    """Compute 3D gaussians Covariance."""

    @staticmethod
    def forward(
        ctx,
        scales,
        quats,
        glob_scale
    ):

        Cov3Ds = _C.compute_3d_covariance_forward(
            scales,
            glob_scale,
            quats
        )

        # Save non-tensors.
        ctx.glob_scale = glob_scale

        # Save tensors.
        ctx.save_for_backward(
            scales,
            quats,
        )

        return Cov3Ds

    @staticmethod
    def backward(
        ctx,
        dL_dCov3Ds
    ):
        (
            scales,
            quats,
        ) = ctx.saved_tensors

        (dL_dscales, 
         dL_drotations) = _C.compute_3d_covariance_backward(
            scales,
            ctx.glob_scale,
            quats,
            dL_dCov3Ds
        )

        # Return a gradient for each input.
        return (
            # scales: Float[Tensor, "*batch 3"],
            dL_dscales,
            # quats: Float[Tensor, "*batch 4"],
            dL_drotations,
            # glob_scale: float,
            None
        )

#---------------------------------------------------------------------#
# Define the C++/CUDA Compute 4D gaussians covariance class and API   #
#---------------------------------------------------------------------#

def compute_4d_gaussians_covariance(
    scales,
    scale_ts,
    quats,
    quat_rs,
    glob_scale = 1.,
):
    """This function computes the 4D gaussians covariance.

    Note:
        This function is differentiable w.r.t the means  and cov3d inputs.

    Args:
       means_grad (Tensor): xyzs grad of gaussians.
       cov3d (Tensor): 3D covariances of gaussians.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor}:

        - **cov4d** (Tensor): 4D covariances.    
        - **mean4d** (Tensor): 4D means.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    return _Compute4DCovariance.apply(
        scales,
        scale_ts,
        quats,
        quat_rs,
        glob_scale
    )

class _Compute4DCovariance(torch.autograd.Function):
    """Compute 4D gaussians Covariance."""

    @staticmethod
    def forward(
        ctx,
        scales,
        scale_ts,
        quats,
        quat_rs,
        glob_scale
    ):

        (
            Cov3Ds,
            Cov_ts,
            speed
        ) = _C.compute_4d_covariance_forward(
            scales,
            scale_ts,
            glob_scale,
            quats,
            quat_rs
        )

        # Save non-tensors.
        ctx.glob_scale = glob_scale

        # Save tensors.
        ctx.save_for_backward(
            scales,
            scale_ts,
            quats,
            quat_rs
        )

        return (Cov3Ds, Cov_ts, speed)

    @staticmethod
    def backward(
        ctx,
        dL_dCov3Ds,
        dL_dCov_ts,
        dL_dspeed
    ):
        (
            scales,
            scale_ts,
            quats,
            quat_rs
        ) = ctx.saved_tensors

        (dL_dscales, 
         dL_dscale_ts, 
         dL_drotations, 
         dL_drotations_r) = _C.compute_4d_covariance_backward(
            scales,
            scale_ts,
            ctx.glob_scale,
            quats,
            quat_rs,
            dL_dCov3Ds,
            dL_dCov_ts,
            dL_dspeed
        )

        # Return a gradient for each input.
        return (
            # scales: Float[Tensor, "*batch 3"],
            dL_dscales,
            # scale_ts: Float[Tensor, "*batch 1"],
            dL_dscale_ts,
            # quats: Float[Tensor, "*batch 4"],
            dL_drotations,
            # quat_rs: Float[Tensor, "*batch 4"],
            dL_drotations_r,
            # glob_scale: float,
            None
        )

#-------------------------------------------------------------#
# Define the C++/CUDA 3D gaussians projection class and API   #
#-------------------------------------------------------------#

def project_gaussians(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 6]
    viewmats: Tensor,  # [4, 4]
    Ks: Tensor,  # [3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = True,
):
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       viewmat (Tensor): view matrix for rendering.
       Ks (float): 3x3 intrinsic matrix.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (4, 4), viewmats.size()
    assert Ks.size() == (3, 3), Ks.size()
    assert covars.size() == (N, 6), covars.size()
    means = means.contiguous()
    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    covars = covars.contiguous()
    return _ProjectGaussians.apply(
        means,
        covars,
        viewmat,
        Ks,
        width,
        height,
        eps2d
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
    )

class _ProjectGaussians(torch.autograd.Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6]
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        radii, means2d, depths, conics, compensations = _C.project_gaussians_forward(
            means,
            covars,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )

       if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            covars,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_viewmats =_C.project_gaussians_backward(
            means,
            covars,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[2],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

