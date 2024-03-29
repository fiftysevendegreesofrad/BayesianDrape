# This file is a derivative work of https://github.com/sbarratt/torch_interpolations
# and hence licensed under Apache 2.0, (c) Shane Barratt and Cardiff University

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
        dist_left[both_zero] = dist_right[both_zero] = 1. # should never happen but treat as equal distance
        return idx_left,idx_right,dist_left,dist_right

    def get_tile_corner_zs(self,interp_xs,interp_ys):
        idx_west,idx_east,dist_west,dist_east = self.dimension_values_distances(interp_xs,self.xs)
        idx_south,idx_north,dist_south,dist_north = self.dimension_values_distances(interp_ys,self.ys)
        
        southwest = self.zs[idx_west,idx_south]
        northwest = self.zs[idx_west,idx_north]
        northeast = self.zs[idx_east,idx_north]
        southeast = self.zs[idx_east,idx_south]
        
        return (southwest,northwest,northeast,southeast),(dist_north,dist_east,dist_south,dist_west)
        
    def single_interpolate(self, interp_xs, interp_ys):
        (southwest,northwest,northeast,southeast),(dist_north,dist_east,dist_south,dist_west) = self.get_tile_corner_zs(interp_xs,interp_ys)
        numerator = southwest*dist_north*dist_east + northwest*dist_south*dist_east + northeast*dist_south*dist_west + southeast*dist_north*dist_west
        denominator = (dist_west+dist_east)*(dist_north+dist_south)
        return numerator / denominator 
        
    def get_max_tile_DZs_single(self,interp_xs,interp_ys):
        corners,_ = self.get_tile_corner_zs(interp_xs,interp_ys)
        corners = torch.hstack([torch.reshape(corner,(-1,1)) for corner in corners])
        res = torch.max(corners,1).values-torch.min(corners,1).values
        assert res.size()==interp_xs.size()
        return res
    
    def get_max_tile_DZs(self,interp_xs,interp_ys,smooth):
        if smooth==0:
            return self.get_max_tile_DZs_single(interp_xs,interp_ys)
        else:
            return 0.25*(self.get_max_tile_DZs_single(interp_xs-smooth,interp_ys-smooth)\
                         +self.get_max_tile_DZs_single(interp_xs+smooth,interp_ys-smooth)\
                         +self.get_max_tile_DZs_single(interp_xs-smooth,interp_ys+smooth)\
                         +self.get_max_tile_DZs_single(interp_xs+smooth,interp_ys+smooth))
        
    def forward(self, interp_xs, interp_ys, smooth):
        if smooth==0:
            return self.single_interpolate(interp_xs,interp_ys)
        else:
            return 0.25*(self.single_interpolate(interp_xs-smooth,interp_ys-smooth)\
                         +self.single_interpolate(interp_xs+smooth,interp_ys-smooth)\
                         +self.single_interpolate(interp_xs-smooth,interp_ys+smooth)\
                         +self.single_interpolate(interp_xs+smooth,interp_ys+smooth))
