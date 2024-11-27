# mixplat

mixplat is an open-source library built on top of [gsplat](https://github.com/nerfstudio-project/gsplat/tree/main) for CUDA-accelerated mixed rasterization of [3D](https://github.com/graphdeco-inria/gaussian-splatting) and [4D](https://github.com/fudan-zvg/4d-gaussian-splatting) gaussians, with Python bindings provided. It supports generating alpha images and inverse depth images. Additionally, it offers support for rendering Gaussians with [hierarchical structures](https://github.com/graphdeco-inria/hierarchical-3d-gaussians).

## Installation
Please install [Pytorch](https://pytorch.org/get-started/previous-versions/) before installing mixplat.

Install the latest commit from GitHub:
```
pip install git+https://github.com/seasideGargantua/mixplat.git
```
## Usage
mixplat divide the entire rendering process into the following steps: 
1. Calculating the covariance matrix
2. Computing the mean, opacity and color
3. Projecting
4. Rasterizing
### Calculating the covariance matrix
First, compute the covariance matrix of the 3D or 4D Gaussian, during which the 4D Gaussian is sliced into 3D.
```
from mixplat.projection import compute_3d_gaussians_covariance, compute_4d_gaussians_covariance

if is_3dgs:
    cov3ds = compute_3d_gaussians_covariance(scale, rotation)
elif is_4dgs:
    cov3ds, cov_t, speed = compute_4d_gaussians_covariance(scale, scale_t, rotation_l, rotation_r)
```
### Computing the mean, opacity and color
The mean and opacity of the 4D Gaussian sliced into 3D space are time-dependent, so they need to be handled separately to facilitate the rendering of the 3D Gaussian mixture. The mixplat provides two methods for calculating colors: one using time-independent spherical harmonics coefficients for static colors, and another using time-dependent spherical harmonics coefficients for dynamic colors.
```
from mixplat.sh import spherical_harmonics_3d, spherical_harmonics_3d_fast, spherical_harmonics_4d

if is_3dgs:
    xyz = _xyz
    opacity = opacity_activation(_opacity)
elif is_4dgs:
    dt = timestamp - _t
    delta_xyz = speed * dt
    xyz = _xyz + delta_xyz
    tshift = 0.5 * dt * dt / cov_t
    opacity = opacity_activation(_opacity) * torch.exp(-tshift)

viewdirs = xyzdetach() - camera_translation
viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)

if rgb_with_t:
    tdirs = _t.detach() - timestamp
    rgb = spherical_harmonics_4d(degree, degree_t, features, viewdirs, tdirs, time_duration)
else:
    rgb = spherical_harmonics_3d_fast(degree, viewdirs, features)
```
### Projecting
After calculating the covariance matrices for the 3D and 4D Gaussians, project them onto a two-dimensional plane.
```
from mixplat.projection import project_gaussians

(xys,
 depths, 
 radii, 
 conics,
 compensation, 
 num_tiles_hit) 
= project_gaussians(
                    xyz,
                    cov3ds,
                    viewmat[:3, :],
                    fx,
                    fy,
                    cx,
                    cy,
                    H,
                    W,
                    16,
                )
```
### Rasterizing
Finally, through rasterization, the final rendered image can be obtained. Currently, mixplat supports rendering three types of images: RGB, alpha, and inverse depth.
```
from mixplat.rasterization import rasterize_gaussians

if return_rgb and return_alpha and return_invdepth:
    rendered_image, rendered_alpha, invdepth = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                rgbs,
                opacity,
                H,
                W,
                16,
                background=bg_color,
                return_alpha=True,
                return_invdepth=True,
            )
elif return_rgb and return_alpha and not return_invdepth:
    rendered_image, rendered_alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                rgbs,
                opacity,
                H,
                W,
                16,
                background=bg_color,
                return_alpha=True,
                return_invdepth=False,
            )
elif return_rgb and not return_alpha and return_invdepth:
    rendered_image, invdepth = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                rgbs,
                opacity,
                H,
                W,
                16,
                background=bg_color,
                return_alpha=False,
                return_invdepth=True,
            )
elif return_rgb and not return_alpha and not return_invdepth:
    rendered_image = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                rgbs,
                opacity,
                H,
                W,
                16,
                background=bg_color,
                return_alpha=False,
                return_invdepth=False,
            )
```
To render Gaussians with a hierarchical structure, you only need to add kid_nodes and interp_weights during the rasterization stage.
```
rendered_image, rendered_alpha = rasterize_gaussians(  # type: ignore
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                rgbs_all,
                opacity_all,
                H,
                W,
                16,
                background=bg_color,
                interp_weights=interpolation_weights,
                kid_nodes=num_node_kids,
                return_alpha=True,
            )
```
## Coordinate
Our coordinate system is the same as [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio). If you are using the camera from 3dgs, please perform the following steps first.
```
R = torch.tensor(viewpoint_camera.R).float().cuda() # 3 x 3
T = torch.tensor(viewpoint_camera.T).float().unsqueeze(1).cuda()  # 3 x 1
w2c = torch.eye(4, device=R.device, dtype=R.dtype)
w2c[:3, :3] = R.T
w2c[:3, 3:4] = T
c2w = w2c.inverse()
c2w[0:3, 1:3] *= -1
# flip the z and y axes to align with nerfstudio conventions
R_edit = torch.diag(torch.tensor([1, -1, -1], dtype=torch.float32)).cuda()
R_new = c2w[:3, :3] @ R_edit
T_new = c2w[:3, 3:4]
# analytic matrix inverse to get world2camera matrix
R_inv = R_new.T
T_inv = -R_inv @ T_new
viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
viewmat[:3, :3] = R_inv
viewmat[:3, 3:4] = T_inv
```

## Credits
Using the algorithm and improvements from:
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) for the main Gaussian Splatting algorithm.
- [4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)  for the 4D Gaussian Splatting algorithm.
- [fast-gaussian-rasterization](https://github.com/dendenxu/fast-gaussian-rasterization) for the fast 3d spherical harmonics and 4d spherical harmonics.
- [gsplat](https://github.com/nerfstudio-project/gsplat) for the fast rendering code base.

## Citation
If you find this code useful, please be so kind to cite
```
@misc{mixplat,  
    title = {Mix Gaussian Splatting},
    howpublished = {GitHub},  
    year = {2024},
    url = {https://github.com/seasideGargantua/mixplat}
}
```
