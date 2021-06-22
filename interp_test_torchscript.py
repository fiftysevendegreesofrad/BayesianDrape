from terrain_interpolator import TerrainInterpolator
from torch_interpolations import RegularGridInterpolator as testinterp
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn
from typing_extensions import Final

class mymodule(torch.nn.Module):

    # do we need to wrap xs,ys,zs as parameters?
    #terrain_interpolator_internal: RegularGridInterpolator
    
    def __init__(self,xs,ys,zs):
        super(mymodule,self).__init__()
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.terrain_interpolator_internal = TerrainInterpolator(self.xs,self.ys, self.zs)
        
    def forward(self,points_to_interpolate,repeats:int=1):
        res = self.terrain_interpolator_internal(points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous())
        for _ in range(repeats-1):
            self.terrain_interpolator_internal(points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous())
        return res
        
        
xs = torch.from_numpy(np.arange(1000))
ys = torch.from_numpy(np.arange(1000))
zs = torch.from_numpy(np.random.random((1000,1000)))

terrain_interpolator_orig = testinterp((xs,ys), zs)

m = mymodule(xs,ys,zs)
m = torch.jit.script(m)

#points_to_interpolate = torch.from_numpy(np.array([[-1,-1],[0,0],[500,500],[500.5,500.5],[999,999],[1000,1000]]))
points_to_interpolate = torch.from_numpy(np.random.random((10000,2))*1000)
old = terrain_interpolator_orig([points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous()])
new = m.forward(points_to_interpolate)


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=True) as prof:
    m.forward(data,1000)
    
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=50))
