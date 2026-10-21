import torch


class TorchInterpolator:
    """Interpolator object that provides one-dimensional interpolation between monotonically
    increasing sample points, with extrapolation beyond sample points.

    Args:
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        method: How to interpolate between points - must be linear for now
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.
        """
    def __init__(self, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, method: str='linear', extrapolate: str='constant'):
        self.dim = dim
        self.xp = xp.movedim(self.dim, -1)
        self.fp = fp.movedim(self.dim, -1)
        self.method = method
        self.extrapolate = extrapolate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the one-dimensional piecewise linear interpolant evaluated at :math:`x`.

        Args:
            x: The :math:`x`-coordinates at which to evaluate the interpolated
                values.

        Returns:
            The interpolated values, same size as `x`.
        """
        self.interpolate(x)

    def interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the one-dimensional piecewise linear interpolant evaluated at :math:`x`.

        Args:
            x: The :math:`x`-coordinates at which to evaluate the interpolated
                values.

        Returns:
            The interpolated values, same size as `x`.
        """
        # Move the interpolation dimension to the last axis
        x = x.movedim(self.dim, -1)
        
        m = torch.diff(self.fp) / torch.diff(self.xp) # slope
        b = self.fp[..., :-1] - m * self.xp[..., :-1] # offset
        indices = torch.searchsorted(self.xp, x, right=False)
        
        if self.extrapolate == 'constant':
            # Pad m and b to get constant values outside of xp range
            m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
            b = torch.cat([self.fp[..., :1], b, self.fp[..., -1:]], dim=-1)
        else: # extrapolate == 'linear'
            indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

        values = m.gather(-1, indices) * x + b.gather(-1, indices)
        
        return values.movedim(-1, self.dim)