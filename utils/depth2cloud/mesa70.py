import torch


class Mesa70PixelGridCloud:
    """Pixel-grid pseudo-point cloud for cameras without depth.

    Returns normalized image-plane coordinates in place of a 3D point cloud.
    x ∈ [-1, 1], y ∈ [-1, 1], z = 0 for all pixels.  The model still gets
    meaningful 2D spatial structure via 3D RoPE; z=0 is a learnable prior.

    The depth / extrinsics / intrinsics arguments are ignored — kept for API
    compatibility with RLBenchDepth2Cloud.
    """

    def __init__(self, shape):
        h, w = shape
        xs = torch.linspace(-1.0, 1.0, w)
        ys = torch.linspace(-1.0, 1.0, h)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) each
        zz = torch.zeros_like(xx)
        self.grid = torch.stack([xx, yy, zz], dim=0)  # (3, H, W)

    def __call__(self, depth, extrinsics, intrinsics):
        """
        depth      : (B, Nc, H, W)  — ignored
        extrinsics : (B, Nc, 4, 4)  — ignored
        intrinsics : (B, Nc, 3, 3)  — ignored
        returns    : (B, Nc, 3, H, W) pixel-grid coords
        """
        b, nc, h, w = depth.shape
        device = depth.device
        grid = self.grid.to(device=device, dtype=depth.dtype)
        if grid.shape[-2] != h or grid.shape[-1] != w:
            import torch.nn.functional as F
            grid = F.interpolate(
                grid.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
        return grid.unsqueeze(0).unsqueeze(0).expand(b, nc, -1, -1, -1)
