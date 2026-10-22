# torchinterp
Adds 1-d interpolation of arbitrarily-sized data to pytorch

Based on https://github.com/pytorch/pytorch/issues/50334#issuecomment-2304751532, but with a callable interpolator object for repeated use and efficient precalculation

# Installation
pip install torchinterp

# Usage

```
xp = torch.tensor([0.0,1.0])
fp = torch.tensor([1.0,2.0])

interpObj = TorchInterpolator(xp,fp,extrapolate='linear')

interpObj(torch.tensor([0.5,0.7]))
```