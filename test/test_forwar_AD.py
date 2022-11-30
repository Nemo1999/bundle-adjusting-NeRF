import torch
import torch.autograd.forward_ad as fwAD


grid = torch.full((5,5),3.).float()
def model(i):
    return torch.nn.functional.grid_sample(grid[None,None,...], i[None,None,None,:], mode="bilinear", align_corners=False)

coord = torch.tensor([0.3,0.3]).float()
tangent = torch.zeros(2).float()

with fwAD.dual_level():
    dual_coord= fwAD.make_dual(coord, tangent)
    assert fwAD.unpack_dual(dual_coord).tangent is tangent
    dual_out = model(dual_coord)
    jvp = fwAD.unpack_dual(dual_out).tangent
print(jvp)




print(model(coord))