import torch


class TorchInterpolator:
    """Interpolator object that provides N-dimensional interpolation between monotonically
    increasing sample points, with extrapolation beyond sample points.

    Args:
        xp: The :math:`x`-coordinates of the data points, must be increasing and 1d
        fp: The :math:`y`-coordinates of the data points; the "dim" dimension must match the size of xp.
        dim: The dimension of y to interpolate; default is 0, so that the *first* dimension of y must match the size of xp.
        method: How to interpolate between points. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'nearest': Uses the nearest existing value. Nearest forces `extrapolate` to behave as 'constant' (for obvious reasons). If multiple values are nearest, will return the first one
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.
        """
    def __init__(self, xp: torch.Tensor, fp: torch.Tensor, dim=0, method: str='linear', extrapolate: str='linear'):
        self.dim = dim
        self.xp = xp.squeeze()
        if self.xp.ndim != 1:
            raise ValueError("Expected an xp tensor that can be squeezed to 1D")
        if torch.any(self.xp.diff() <= 0):
            raise ValueError("Expected a monotonically increasing xp tensor")
        if fp.ndim == 1:
            self.fp = fp
            self.singleDimOutput = True
            if (self.fp.shape != self.xp.shape):
                raise ValueError("fp size incompatible with xp size; fp[dim] and xp must match")
        else:
            self.fp = fp.movedim(dim,-1)
            self.singleDimOutput = False
            if (self.fp.shape[-1] != self.xp.shape[-1]):
                raise ValueError("fp size incompatible with xp size; fp[dim] and xp must match")
        self.method = method
        if (method == 'nearest'):
            self.extrapolate = 'constant'
        else:
            self.extrapolate = extrapolate

        if self.method == 'linear':
            #slopetensor's last dimension should be the length of xp minus 1, and its other dimensions should be the size of the data dimensions of fp
            #offsetTensor's last dimension should be the length of xp minus 1, and its other dimensions should be the size of the data dimensions of fp
            self.slopeTensor = torch.diff(self.fp) / torch.diff(self.xp)
            if self.singleDimOutput:
                self.offsetTensor = self.fp[:-1] - self.slopeTensor * self.xp[:-1]
                def infer_val(x_i: torch.Tensor, index_i: torch.Tensor) -> torch.Tensor:
                    return self.slopeTensor[index_i] * x_i + self.offsetTensor[index_i]
            else:
                self.offsetTensor = self.fp[..., :-1] - self.slopeTensor * self.xp[:-1]
                def infer_val(x_i: torch.Tensor, index_i: torch.Tensor) -> torch.Tensor:
                    return self.slopeTensor[...,index_i] * x_i + self.offsetTensor[...,index_i]

            #witchcraft going on here. Although theoretically we could leave out_dims at 0 and
            #omit the movedim happening in the return, because we let the user specify the interpolation dimension of y,
            #we have to do both ops.
            self.mapped_infer = torch.vmap(infer_val,in_dims=(0,0),out_dims=-1)
        else:
            if self.singleDimOutput:
                def infer_val(index_i: torch.Tensor) -> torch.Tensor:
                    return self.fp[index_i]
            else:
                def infer_val(index_i: torch.Tensor) -> torch.Tensor:
                    return self.fp[...,index_i]
            #witchcraft going on here. Although theoretically we could leave out_dims at 0 and
            #omit the movedim happening in the return, because we let the user specify the interpolation dimension of y,
            #we have to do both ops.
            self.mapped_infer = torch.vmap(infer_val,in_dims=0,out_dims=-1)


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the one-dimensional piecewise linear interpolant evaluated at :math:`x`.

        Args:
            x: The :math:`x`-coordinates at which to evaluate the interpolated
                values.

        Returns:
            The interpolated values, same size as `x`.
        """
        return self.interpolate(x)

    def interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the one-dimensional piecewise linear interpolant evaluated at :math:`x`.

        Args:
            x: The :math:`x`-coordinates at which to evaluate the interpolated
                values.

        Returns:
            The interpolated values, same size as `x`.
        """
        singleInference = x.ndim == 0
        x = torch.atleast_1d(x.squeeze())
        if x.ndim != 1:
            raise ValueError("Expected a tensor that can be squeezed to 1D")

        if (self.method == 'linear'):
            #clamp the input to the extremes of the data set in order to do constant extrapolation
            if (self.extrapolate =='constant'):
                x = x.clamp(min=self.xp[0], max=self.xp[-1])

            indices = torch.searchsorted(self.xp, x, right=False)-1
            indices = indices.clamp(min=0, max=self.slopeTensor.shape[-1]-1)
            
            result = self.mapped_infer(x, indices)
        else:
            #each entry in diffrows is a minimizable tensor of time differences
            diffRows = torch.abs(x.unsqueeze(1)-self.xp.unsqueeze(0))
            #each index in indices is the closest time-index over xp for each time in x
            indices = torch.argmin(diffRows, dim=-1, keepdim=False)
            result = self.mapped_infer(indices)

        if (not self.singleDimOutput):
            result = result.movedim(-1,self.dim)
        if singleInference:
            result = result.squeeze()
        return result