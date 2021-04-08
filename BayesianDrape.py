# -*- coding: utf-8 -*-

"""
***************************************************************************
*   Copyright Crispin Cooper 2021                                         *
***************************************************************************
"""

from optparse import OptionParser
import rioxarray # gdal raster is another option should this give trouble though both interpolate with scipy underneath
from xarray import DataArray
from scipy.optimize import minimize
from scipy.stats import norm,expon
from scipy.spatial import distance_matrix
from scipy.sparse import lil_matrix
from scipy.interpolate import RegularGridInterpolator,interp1d
import geopandas as gp
import pandas as pd
import numpy as np
from ordered_set import OrderedSet
from itertools import tee
from shapely.geometry import LineString
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
    
op = OptionParser()
op.add_option("--TERRAIN-INPUT",dest="terrainfile",help="[REQUIRED] Terrain model",metavar="FILE")
op.add_option("--POLYLINE-INPUT",dest="shapefile",help="[REQUIRED] Polyline feature class e.g. network or GPS trace",metavar="FILE")
op.add_option("--OUTPUT",dest="outfile",help="[REQUIRED] Output feature class",metavar="FILE")
op.add_option("--SLOPE-PRIOR-STD",dest="slope_prior_std",help="[REQUIRED] Standard deviation of zero-centred prior for path slope",metavar="ANGLE_IN_DEGREES",type="float")
op.add_option("--SPATIAL-MISMATCH-PRIOR-STD",dest="mismatch_prior_std",help="[REQUIRED] Standard deviation of zero-centred prior for spatial mismatch (in spatial units of projection)",metavar="DISTANCE",type="float")
op.add_option("--JUST-LLTEST",dest="just_lltest",action="store_true",help="Test mode")
(options,args) = op.parse_args()

if options.just_lltest:
    options.terrainfile = "data/all_os50_terrain.tif"
    options.shapefile = "data/test_awkward_link.shp"
    options.slope_prior_std = 2.2
    options.mismatch_prior_std = 25
else:
    missing_options = []
    for option in op.option_list:
        if option.help.startswith(r'[REQUIRED]') and eval('options.' + option.dest) == None:
            missing_options.extend(option._long_opts)
    if len(missing_options) > 0:
        op.error('Missing REQUIRED parameters: ' + str(missing_options))

net_df = gp.read_file(options.shapefile)
terrain_raster = rioxarray.open_rasterio(options.terrainfile)

print (f"{net_df.crs=}\nterrain raster crs??")
# todo assert these projections are the same

terrain_xs = np.array(terrain_raster.x,np.float64)
terrain_ys = np.flip(np.array(terrain_raster.y,np.float64)) 
terrain_data = np.flip(terrain_raster.data[0],axis=0).T # fixme correct?
terrain_interpolator = RegularGridInterpolator((terrain_xs,terrain_ys), terrain_data)
del terrain_xs, terrain_ys, terrain_data, terrain_raster

# Todo: add points to linestrings where they cross raster cells or at least every cell-distance
# (latter may make more sense as more expensive to compute gradients at cell boundaries)

# Represent points independently of linestrings:
# - and represent each linestring with a point index list 
# xs, ys initially stored in OrderedSet so we can match line endpoints; later converted to array
# If this gets too slow we can potentially do dict lookup for line endpoints only

all_points_set = OrderedSet()
point_indices_all_rows = []

for _,row in net_df.iterrows():
    assert row.geometry.geom_type=='LineString'
    xs,ys = row.geometry.coords.xy
    point_indices = []
    for x,y in zip(xs,ys):
        index = all_points_set.add((x,y))
        point_indices.append(index)
    point_indices_all_rows.append(point_indices)
net_df["point_indices"] = point_indices_all_rows # fixme what if column name exists already?

# Build point adjacency matrix

num_points = len(all_points_set)
print (f"{num_points=}")
adjacency = lil_matrix((num_points,num_points),dtype=bool) 

for _,row in net_df.iterrows():
    xs,ys = row.geometry.coords.xy
    for point1,point2 in pairwise(zip(xs,ys)):
        index1 = all_points_set.index(point1)
        index2 = all_points_set.index(point2)
        adjacency[index1,index2]=True
        # provided we optimize all points together, we don't store the reverse adjacency 
        # otherwise each gradient likelihood is counted twice, breaking log likelihood
        # if we did want to compute gradient for a single point, we should compute reverse adjacency and also halve all gradient log likelihoods
        # adjacency[index2,index1]=True 
        
all_points = np.array(all_points_set)
del all_points_set
adjacency = adjacency.tocsr().toarray() # undo sparseness for now

# Define posterior log likelihood

grade_mean = np.tan(options.slope_prior_std*np.pi/180)
grade_exp_dist_lambda = 1/grade_mean
log_grade_exp_dist_lambda = np.log(grade_exp_dist_lambda)
def grade_logpdf(x):
    return log_grade_exp_dist_lambda - grade_exp_dist_lambda*x

#test_slope_angles = np.arange(90)
#test_grades = np.arctan(test_slope_angles/180*np.pi)
#test_pdf = np.exp(grade_logpdf(test_grades))
#print(f"{test_slope_angles=}\n{test_pdf=}")

max_displacement=100
dist_range = np.arange(100)/20*max_displacement # todo take from command line, also interpolation array size
offset_logpdf = norm(scale=options.mismatch_prior_std).logpdf
approxlognorm_from_squaredist = interp1d(dist_range**2,offset_logpdf(dist_range),bounds_error=False,fill_value=-np.inf) # doing this reduced looking up normal pdf from 35% to 10% of runtime of minus_log_likelihood

# wrap interpolators in functions for profiling stats
def approxlognormf(x):
    return approxlognorm_from_squaredist(x)
def terrain_interpolatorf(x):
    return terrain_interpolator(x)

inv_distances = (distance_matrix(all_points,all_points)+np.eye(num_points))**-1 # fixme optimise sparse only compute for neighbours
llcount=0
def minus_log_likelihood(point_offsets,adjacency=adjacency):
    global llcount
    llcount += 1
    point_offsets = point_offsets.reshape(all_points.shape) # as optimize flattens point_offsets
    points_to_interpolate = all_points + point_offsets
    zs = terrain_interpolatorf(points_to_interpolate)
    neighbour_grades = adjacency * abs(zs[:, None] - zs[None, :]) * inv_distances
    neighbour_likelihood = grade_logpdf(neighbour_grades).sum() # fixme this currently includes non-neighbours as constant, can we fix with sparse?
    offset_square_distances = ((point_offsets**2).sum(axis=1))
    offset_likelihood = approxlognormf(offset_square_distances).sum()
    return -(neighbour_likelihood+offset_likelihood)

# Test function

if options.just_lltest:
    offset_unit_vector = np.zeros((num_points,2),float)
    
    from timeit import Timer
    ncalls = 30
    nrepeats = 10
    t = Timer(lambda: minus_log_likelihood(offset_unit_vector))
    print("Current impl time:",min(t.repeat(number=ncalls,repeat=nrepeats)))
    if hasattr(adjacency,"toarray"):
        nonsparse_adj = adjacency.toarray()
        t = Timer(lambda: minus_log_likelihood(offset_unit_vector,adjacency=nonsparse_adj))
        print("Nonsparse time:",min(t.repeat(number=ncalls,repeat=nrepeats)))
    
    
    for i in range(num_points):
        offset_unit_vector[i]=np.array([(i//3)%3,i%3])-1
    original_lls = [29233.79119917465, 29869.792894256196, 30479.836034098767, 31233.911375847896, 32133.64223632107]
    new_lls = []
    for i,oll in enumerate(original_lls):
        ll = minus_log_likelihood(offset_unit_vector*i)
        new_lls.append(ll)
        passed=(ll==oll)
        print (f"{i=} {oll=} {ll=} {passed=}")
    print (f"{new_lls=}")
        
    sys.exit(0)

# Run optimizer

class TerminateOptException(Exception):
    pass

def callback(x):
    global llcount,last_ll,result
    ll = -minus_log_likelihood(x)
    lldiff = abs(ll-last_ll)
    last_ll = ll
    print (f"callback {llcount=} {ll=} {lldiff=}")
    llcount=0
    if lldiff<1: #fixme magic number 
        result = x # yuck
        raise TerminateOptException()
    
result = np.zeros((num_points,2),float)
initial_log_likelihood = -minus_log_likelihood(result)
last_ll = initial_log_likelihood
try:
    result = minimize(minus_log_likelihood,result,callback = callback) 
except TerminateOptException:
    pass
result = result.reshape(all_points.shape)
final_points = all_points + result
optimal_zs = terrain_interpolator(final_points)
final_log_likelihood = -minus_log_likelihood(result)

# Print result report

offset_distances = ((result**2).sum(axis=1))**0.5
mean_offset_dist = offset_distances.mean()
max_offset_dist = offset_distances.max()
print (f"{initial_log_likelihood=}\n{final_log_likelihood=}\n{mean_offset_dist=}\n{max_offset_dist=}")

# Reconstruct geometries
xs = all_points[:,0]
ys = all_points[:,1]
net_df.geometry = net_df.apply(lambda row: LineString([(xs[pt_index],ys[pt_index],optimal_zs[pt_index]) for pt_index in row.point_indices]),axis=1)
del net_df["point_indices"]

# Write output
 
net_df.to_file(options.outfile)
