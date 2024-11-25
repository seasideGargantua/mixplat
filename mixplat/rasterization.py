import torch
import mixplat.cuda as _C
from .utils import compute_cumulative_intersects, bin_and_sort_gaussians

#------------------------------------------------------------#
# Define the C++/CUDA rasterization class and API            #
#------------------------------------------------------------#

def rasterize_gaussians(
    xys,
    depths,
    radii,
    conics,
    num_tiles_hit,
    colors,
    opacity,
    img_height,
    img_width,
    block_width,
    background = None,
    interp_weights = None,
    kid_nodes = None,
    return_alpha = False,
    return_invdepth = False,
):
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call. integer number of pixels between 2 and 16 inclusive
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel
        return_invdepth (bool): whether to return inverse depth channel

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
        - **out_invdepth** (Optional[Tensor]): Inverse depth channel of the rendered output image.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if background is not None:
        assert (
            background.shape[0] == colors.shape[-1]
        ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
    else:
        background = torch.ones(
            colors.shape[-1], dtype=torch.float32, device=colors.device
        )

    if interp_weights is None:
        interp_weights = torch.tensor([], device=xys.device)
    if kid_nodes is None:
        kid_nodes = torch.tensor([], device=xys.device, dtype=torch.int32)

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussians.apply(
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        opacity.contiguous(),
        img_height,
        img_width,
        block_width,
        background.contiguous(),
        interp_weights.contiguous(),
        kid_nodes.contiguous(),
        return_alpha,
        return_invdepth,
    )

class _RasterizeGaussians(torch.autograd.Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        img_height,
        img_width,
        block_width,
        background = None,
        interp_weights = None,
        kid_nodes = None,
        return_alpha = False,
        return_invdepth = False,
    ):
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
            )
            out_alpha = (
                torch.zeros(img_height, img_width, device=xys.device)
            )
            out_invdepth = (
                torch.ones(img_height, img_width, device=xys.device)
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )
            rasterize_fn = _C.rasterize_forward

            out_img, out_invdepth, final_Ts, final_idx = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                return_invdepth,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                depths,
                opacity,
                background,
                interp_weights,
                kid_nodes,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.return_invdepth = return_invdepth
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            depths,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
            interp_weights,
            kid_nodes,
        )
        out = (out_img,)
        if return_alpha:
            out_alpha = 1 - final_Ts
            out += (out_alpha,)
        if return_invdepth:
            out += (out_invdepth,)

        if len(out) == 1:
            out = out[0]

        return out

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None, v_out_invdepth=None):
        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])
        if v_out_invdepth is None:
            v_out_invdepth = torch.zeros_like(v_out_img[..., 0])

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            depths,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
            interp_weights,
            kid_nodes,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)
            v_depth = torch.zeros_like(depths)
        else:
            rasterize_fn = _C.rasterize_backward
            v_xy, v_conic, v_colors, v_opacity, v_depth = rasterize_fn(
                img_height,
                img_width,
                ctx.block_width,
                ctx.return_invdepth,
                interp_weights,
                kid_nodes,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                depths,
                opacity,
                background,
                final_Ts,
                final_idx,
                v_out_img,
                v_out_alpha,
                v_out_invdepth,
            )

        return (
            v_xy,  # xys
            v_depth,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            None,  # img_height
            None,  # img_width
            None,  # block_width
            None,  # background
            None,  # interp_weights
            None,  # kid_nodes
            None,  # return_alpha
            None,  # return_invdepth
        )
