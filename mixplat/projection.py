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
    means3d,
    cov3d,
    viewmat,
    fx,
    fy,
    cx,
    cy,
    img_height,
    img_width,
    block_width = 16,
    clip_thresh = 0.01
):
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
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
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    return _ProjectGaussians.apply(
        means3d.contiguous(),
        cov3d.contiguous(),
        viewmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh
    )

class _ProjectGaussians(torch.autograd.Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d,
        cov3d,
        viewmat,
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh = 0.01
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            cov3d,
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            cov3d,
            viewmat,
            radii,
            conics,
            compensation,
        )

        return (xys, depths, radii, conics, compensation, num_tiles_hit)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_depths,
        v_radii,
        v_conics,
        v_compensation,
        v_num_tiles_hit
    ):
        (
            means3d,
            cov3d,
            viewmat,
            radii,
            conics,
            compensation,
        ) = ctx.saved_tensors

        (v_cov2d, v_cov3d, v_mean3d, v_viewmat) = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            viewmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            compensation,
            v_xys,
            v_depths,
            v_conics,
            v_compensation,
        )

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # cov3d: Float[Tensor, "*batch 3 3"],
            v_cov3d,
            # viewmat: Float[Tensor, "3 4"],
            v_viewmat,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # block_width: int,
            None,
            #delta_means: Optional[Tensor],
            None,
            # clip_thresh,
            None,
            # mask: Optional[Tensor],
            None,
        )

