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
from terrain_interpolator import TerrainInterpolator
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
                pitch_angle_scale=pitch_prior,use_cuda=False)
    
    num_estimated_points = int(model.initial_guess.shape[0])
    print (f"{num_estimated_points=}")
    error_vector = np.zeros((num_estimated_points),float)
    grad_test_input = torch.flatten(np_to_torch(error_vector))
   
    print ("Old gradient[0]",original_gradient)
    new_gradient_0 = model.minus_log_likelihood_gradient(grad_test_input)[0].numpy()
    print("New gradient[0]",new_gradient_0)
    
    for i in range(num_estimated_points):
        error_vector[i]=(i%3)-1
    
    new_lls = []
    for i,oll in enumerate(original_lls):
        ll = float(model.minus_log_likelihood(torch.flatten(np_to_torch(error_vector*i))))
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
    lltest("data/test_awkward_link.shp",[1510.4459937298175, 2158.157527105684, 2893.2124806343572, 3610.773318141067, 4282.4883512914785],[10.77271613,  2.46512369])

def test_log_likelihood_large():
    lltest("data/biggertest.shp",[45557.20802136228, 69209.13185062712, 98373.28361210058, 126995.76480611488, 154641.15570633655],[-0.42851993,  0.03308218])

def test_autodiff_cell_boundary():
    '''Ensure there is still a gradient on boundaries of raster cells'''
    terrain_raster = rioxarray.open_rasterio("data/all_os50_terrain.tif")
    terrain_xs = np_to_torch(np.array(terrain_raster.x,np.float64))
    terrain_ys = np_to_torch(np.flip(np.array(terrain_raster.y,np.float64)).copy() )
    terrain_data = np_to_torch(np.flip(terrain_raster.data[0],axis=0).T.copy())
    terrain_interpolator_inner = TerrainInterpolator(terrain_xs,terrain_ys, terrain_data)
    
    def print_point_inter_and_grad(point):
        cellboundryx = point[0] in terrain_xs
        cellboundryy = point[1] in terrain_ys
        pt = np_to_torch(np.array([point],dtype=float))
        pt.requires_grad = True
        terr = terrain_interpolator_inner.forward(pt[:,0].contiguous(),pt[:,1].contiguous())
        terr.backward()
        grad = pt.grad
        print (f"{pt=}\n{cellboundryx=} {cellboundryy=}\n{terr=}\n{grad=}\n")
        assert grad.sum()!=0
       
    for point in [(355000,205000),(355025,205025),(355050,205050),(355075,205075)]:
        print_point_inter_and_grad(point)

    
test_log_likelihood_small()