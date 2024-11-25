import torch
import mixplat.cuda as _C
import math

#---------------------------------------------------------------------#
# Define the C++/CUDA Gaussians rasterization utils API               #
#---------------------------------------------------------------------#

def map_gaussian_to_intersects(
    num_points,
    num_intersects,
    xys,
    depths,
    radii,
    cum_tiles_hit,
    tile_bounds,
    block_size,
):
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        block_size,
    )
    return (isect_ids, gaussian_ids)

def get_tile_bin_edges(
    num_intersects,
    isect_ids_sorted,
    tile_bounds,
):
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(
        num_intersects, isect_ids_sorted.contiguous(), tile_bounds
    )

def compute_cumulative_intersects(num_tiles_hit):
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit

def bin_and_sort_gaussians(
    num_points,
    num_intersects,
    xys,
    depths,
    radii,
    cum_tiles_hit,
    tile_bounds,
    block_size,
):
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        tile_bounds,
        block_size,
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins

class RelocationOp:
    """Relocation operator for opacity and scale values.

    This operator takes in the old opacity and scale values and returns the new values after relocation.

    Note:
        This class is not differentiable to any input.

    Args:
        opacity_old (Tensor): old opacity values.
        scale_old (Tensor): old scale values.
        scale_t_old (Tensor, Optional): old scale t values.
        N (Tensor): number of relocations to perform.

    Returns:
        A tuple of {Tensor, Tensor}:
        - **opacity_new** (Tensor): new opacity values.
        - **scale_new** (Tensor): new scale values.
        - **scale_t_new** (Tensor, Optional): new scale t values.
    """
    def __init__(self, N_max=51):
        self.N_max = 51
        self.binoms = torch.zeros((N_max, N_max)).float().cuda()
        for n in range(N_max):
            for k in range(n+1):
                self.binoms[n, k] = math.comb(n, k)

    def compute_relocation(self, opacity_old, scale_old, N, scale_t_old=None):
        if scale_t_old is None:
            scale_t_old = scale_old.clone()[:,0]
        N.clamp_(min=1, max=self.N_max-1)
        return _C.compute_relocation(opacity_old, scale_old, scale_t_old, N, self.binoms, self.N_max)