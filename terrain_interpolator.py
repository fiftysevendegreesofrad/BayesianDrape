import torch
import torch.nn

class TerrainInterpolator(torch.nn.Module):

    def __init__(self, xs, ys, zs):
        super(TerrainInterpolator,self).__init__()

        self.xs = xs
        self.ys = ys
        self.zs = zs
        
        assert len(xs.shape)==1
        assert len(ys.shape)==1
        assert zs.shape == (xs.shape[0],ys.shape[0])

    def dimension_values_distances(self,interp_points, buckets):
        idx_right = torch.bucketize(interp_points, buckets)
        idx_right[idx_right >= buckets.shape[0]] = buckets.shape[0] - 1
        idx_left = (idx_right - 1).clamp(0, buckets.shape[0] - 1)
        dist_left = interp_points - buckets[idx_left]
        dist_right = buckets[idx_right] - interp_points
        dist_left[dist_left < 0] = 0. 
        dist_right[dist_right < 0] = 0.
        both_zero = (dist_left == 0) & (dist_right == 0)
        dist_left[both_zero] = dist_right[both_zero] = 1. # correct? test
        return idx_left,idx_right,dist_left,dist_right

    def forward(self, interp_xs, interp_ys):
        idx_west,idx_east,dist_west,dist_east = self.dimension_values_distances(interp_xs,self.xs)
        idx_south,idx_north,dist_south,dist_north = self.dimension_values_distances(interp_ys,self.ys)
        
        southwest = self.zs[idx_west,idx_south]
        northwest = self.zs[idx_west,idx_north]
        northeast = self.zs[idx_east,idx_north]
        southeast = self.zs[idx_east,idx_south]
        
        numerator = southwest*dist_north*dist_east + northwest*dist_south*dist_east + northeast*dist_south*dist_west + southeast*dist_north*dist_west
        denominator = (dist_west+dist_east)*(dist_north+dist_south)
        
        return numerator / denominator
