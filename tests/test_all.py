import sys,os

if __name__ == '__main__':
    encoding = sys.getfilesystemencoding()
    path = os.path.dirname(str(__file__))
    parentdir = os.path.dirname(path)
    if parentdir not in sys.path:
        sys.path.insert(0,parentdir)

import BayesianDrape
import numpy as np
import geopandas as gp
import torch,rioxarray
from torch_interpolations import RegularGridInterpolator
from timeit import Timer
        
def np_to_torch(x):
    return torch.from_numpy(x)

def lltest(net_file,original_lls,original_gradient,times=False):
    net_df = gp.read_file(net_file)
    terrain_raster = rioxarray.open_rasterio("data/all_os50_terrain.tif")
    terrain_xs = np.array(terrain_raster.x,np.float64)
    terrain_ys = np.array(terrain_raster.y,np.float64) 
    terrain_data = terrain_raster.data[0]
    
    slope_prior = 2.2
    cont_scale = np.arctan(0.42)/np.pi*180
    pitch_prior = 1.3
    model = BayesianDrape.build_model(terrain_xs,terrain_ys,terrain_data,net_df.geometry,slope_prior_scale=slope_prior,slope_continuity_scale=cont_scale,
                pitch_angle_scale=pitch_prior)
    
    num_estimated_points = int(model.initial_guess.shape[0]/2)
    print (f"{num_estimated_points=}")
    offset_unit_vector = np.zeros((num_estimated_points,2),float)
    grad_test_input = torch.flatten(np_to_torch(offset_unit_vector))
   
    print ("Old gradient[0]",original_gradient)
    new_gradient_0 = model.minus_log_likelihood_gradient(grad_test_input)[0:2].numpy()
    print("New gradient[0]",new_gradient_0)
    
    for i in range(num_estimated_points):
        offset_unit_vector[i]=np.array([(i//3)%3,i%3])-1
    
    new_lls = []
    for i,oll in enumerate(original_lls):
        ll = float(model.minus_log_likelihood(torch.flatten(np_to_torch(offset_unit_vector*i))))
        new_lls.append(ll)
        passed=(ll==oll)
        print (f"{i=} {oll=} {ll=} {passed=}")
    print (f"{new_lls=}")
    assert original_lls==new_lls
    
    if times:
        time_number = 50
        t = Timer(lambda: model.minus_log_likelihood_gradient(grad_test_input))
        print("Current gradient time:",min(t.repeat(number=time_number,repeat=3))/time_number)
    

        
def test_log_likelihood_small():
    lltest("data/test_awkward_link.shp",[1510.445993729818, 2158.1575271056854, 2893.212480634359, 3610.7733181410676, 4282.488351291478],[10.77271613,  2.46512369])

def test_log_likelihood_large():
    lltest("data/biggertest.shp",[45557.208021362145, 69209.13185062728, 98373.28361210061, 126995.76480611479, 154641.1557063365],[-0.42851993,  0.03308218])

def test_autodiff_cell_boundary():
    '''Ensure there is still a gradient on boundaries of raster cells'''
    terrain_raster = rioxarray.open_rasterio("data/all_os50_terrain.tif")
    terrain_xs = np_to_torch(np.array(terrain_raster.x,np.float64))
    terrain_ys = np_to_torch(np.flip(np.array(terrain_raster.y,np.float64)).copy() )
    terrain_data = np_to_torch(np.flip(terrain_raster.data[0],axis=0).T.copy())
    terrain_interpolator_inner = RegularGridInterpolator((terrain_xs,terrain_ys), terrain_data)
    
    def print_point_inter_and_grad(point):
        pt = np_to_torch(np.array(point,dtype=float))
        pt.requires_grad = True
        ptr = pt.reshape([2,1])
        terr = terrain_interpolator_inner(ptr)
        terr.backward()
        grad = pt.grad
        print (f"{ptr=}\n{terr=}\n{grad=}")
        assert grad.sum()!=0
        
    for point in [(355000,205000),(355025,205025),(355050,205050),(355075,205075)]:
        print_point_inter_and_grad(point)

    
