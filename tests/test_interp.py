import torch

from torchinterp import TorchInterpolator


def test_linear_single_dim():
    xp = torch.tensor([0.0,1.0])
    fp = torch.tensor([1.0,2.0])
    interpObj = TorchInterpolator(xp,fp,extrapolate='linear')
    assert torch.equal(interpObj(torch.tensor(0.5)),torch.tensor(1.5))
    assert torch.equal(interpObj(torch.tensor([0.5])),torch.tensor([1.5]))
    assert torch.equal(interpObj(torch.tensor([0.5,0.7])),torch.tensor([1.5,1.7]))
    assert torch.equal(interpObj(torch.tensor([0.5,1.7])),torch.tensor([1.5,2.7]))
    assert torch.equal(interpObj(torch.tensor([-0.5,1.7])),torch.tensor([0.5,2.7]))
    interpObj = TorchInterpolator(xp,fp,extrapolate='constant')
    assert torch.equal(interpObj(torch.tensor(1.5)),torch.tensor(2.0))
    assert torch.equal(interpObj(torch.tensor([1.5])),torch.tensor([2.0]))
    assert torch.equal(interpObj(torch.tensor([0.5,1.7])),torch.tensor([1.5,2.0]))
    assert torch.equal(interpObj(torch.tensor([-0.5,1.7])),torch.tensor([1.0,2.0]))

def test_linear_multi_dim():
    xp = torch.tensor([0.0,1.0,3.0])
    fp = torch.tensor([[1.0,1.0],[2.0,3.0],[4.0,7.0]])
    interpObj = TorchInterpolator(xp,fp,extrapolate='linear')
    assert torch.equal(interpObj(torch.tensor(0.5)),torch.tensor([1.5,2.0]))
    assert torch.equal(interpObj(torch.tensor([0.5])),torch.tensor([[1.5,2.0]]))
    assert torch.equal(interpObj(torch.tensor([0.5,2.0])),torch.tensor([[1.5,2.0],[3.0,5.0]]))
    assert torch.equal(interpObj(torch.tensor([0.5,3.7])),torch.tensor([[1.5,2.0],[4.7,8.4]]))
    assert torch.equal(interpObj(torch.tensor([-0.5,3.7])),torch.tensor([[0.5,0.0],[4.7,8.4]]))
    interpObj = TorchInterpolator(xp,fp,extrapolate='constant')
    assert torch.equal(interpObj(torch.tensor(3.5)),torch.tensor([4.0,7.0]))
    assert torch.equal(interpObj(torch.tensor([3.5])),torch.tensor([[4.0,7.0]]))
    assert torch.equal(interpObj(torch.tensor([0.5,3.7])),torch.tensor([[1.5,2.0],[4.0,7.0]]))
    assert torch.equal(interpObj(torch.tensor([-0.5,3.7])),torch.tensor([[1.0,1.0],[4.0,7.0]]))

def test_linear_many_dim():
    xp = torch.tensor([0.0,1.0])
    fp = torch.tensor([[[1.0,0.0],[1.0,1.0]],[[2.0,4.0],[3.0,9.0]]])
    interpObj = TorchInterpolator(xp,fp,extrapolate='linear')
    assert torch.equal(interpObj(torch.tensor(0.5)),torch.tensor([[1.5,2.0],[2.0,5.0]]))

def test_nearest_single_dim():
    xp = torch.tensor([0.0,1.0])
    fp = torch.tensor([1.0,2.0])
    interpObj = TorchInterpolator(xp,fp,method='nearest')
    assert torch.equal(interpObj(torch.tensor(0.5)),torch.tensor(1.0))
    assert torch.equal(interpObj(torch.tensor([0.5])),torch.tensor([1.0]))
    assert torch.equal(interpObj(torch.tensor([0.5,0.7])),torch.tensor([1.0,2.0]))
    assert torch.equal(interpObj(torch.tensor([0.5,1.7])),torch.tensor([1.0,2.0]))
    assert torch.equal(interpObj(torch.tensor([-0.5,1.7])),torch.tensor([1.0,2.0]))

def test_nearest_multi_dim():
    xp = torch.tensor([0.0,1.0,3.0])
    fp = torch.tensor([[1.0,1.0],[2.0,3.0],[4.0,7.0]])
    interpObj = TorchInterpolator(xp,fp,method='nearest')
    assert torch.equal(interpObj(torch.tensor(0.5)),torch.tensor([1.0,1.0]))
    assert torch.equal(interpObj(torch.tensor([0.5])),torch.tensor([[1.0,1.0]]))
    assert torch.equal(interpObj(torch.tensor([0.5,2.1])),torch.tensor([[1.0,1.0],[4.0,7.0]]))
    assert torch.equal(interpObj(torch.tensor([0.5,3.7])),torch.tensor([[1.0,1.0],[4.0,7.0]]))
    assert torch.equal(interpObj(torch.tensor([-0.5,3.7])),torch.tensor([[1.0,1.0],[4.0,7.0]]))

def test_nearest_many_dim():
    xp = torch.tensor([0.0,1.0])
    fp = torch.tensor([[[1.0,0.0],[1.0,1.0]],[[2.0,4.0],[3.0,9.0]]])
    interpObj = TorchInterpolator(xp,fp,method='nearest')
    assert torch.equal(interpObj(torch.tensor(0.6)),torch.tensor([[2.0,4.0],[3.0,9.0]]))