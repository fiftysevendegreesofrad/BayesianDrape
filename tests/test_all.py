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
    terrain_xs = (np.array(terrain_raster.x,np.float64))
    terrain_ys = (np.flip(np.array(terrain_raster.y,np.float64)) )
    terrain_data = (np.flip(terrain_raster.data[0],axis=0).T)
    
    model = BayesianDrape.build_model(terrain_xs,terrain_ys,terrain_data,net_df.geometry)
    
    num_estimated_points = int(model.initial_guess.shape[0]/2)
    print (f"{num_estimated_points=}")
    offset_unit_vector = np.zeros((num_estimated_points,2),float)
    grad_test_input = torch.flatten(np_to_torch(offset_unit_vector))
   
    print ("Old gradient[0]",original_gradient)
    print("New gradient[0]",model.minus_log_likelihood_gradient(grad_test_input)[0:2].numpy())
    
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
    lltest("data/test_awkward_link.shp",[666.3370627327566, 680.6210939337058, 715.9840728425804, 774.6845624725942, 869.3440722132914],[-0.55813296,-0.12771772])
    
def test_log_likelihood_large():
    lltest("data/biggertest.shp",[35556.77348013957, 35842.87091246876, 36532.27095995685, 37558.64256056091, 38872.721038508236],[-0.09709856, 0.00749611])

def test_log_likelihood_time():
    lltest("data/biggertest.shp",[35556.77348013957, 35842.87091246876, 36532.27095995685, 37558.64256056091, 38872.721038508236],[-0.09709856, 0.00749611],True)
    
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

def is_nearly(a,b):
    return abs(a-b)<0.1

def test_grade_change_as_angle():
    assert is_nearly(BayesianDrape.grade_change(0,1),1)
    assert is_nearly(BayesianDrape.grade_change(1,2),abs(np.tan(18.4*np.pi/180)))
    assert is_nearly(BayesianDrape.grade_change(1,-2),abs(np.tan(108.4*np.pi/180)))
    assert is_nearly(BayesianDrape.grade_change(2,-1),abs(np.tan(108.4*np.pi/180)))
    assert is_nearly(BayesianDrape.grade_change(2,0),abs(np.tan(63.4*np.pi/180)))
    
if __name__ == '__main__':
    test_log_likelihood_time()