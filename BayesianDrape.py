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
from scipy.stats import norm
from scipy.spatial import distance_matrix
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

missing_options = []
for option in op.option_list:
    if option.help.startswith(r'[REQUIRED]') and eval('options.' + option.dest) == None:
        missing_options.extend(option._long_opts)
if len(missing_options) > 0:
    op.error('Missing REQUIRED parameters: ' + str(missing_options))

net_df = gp.read_file(options.shapefile)
terrain_raster = rioxarray.open_rasterio(options.terrainfile)

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
adjacency = np.zeros((num_points,num_points),bool) # todo change to sparse

for _,row in net_df.iterrows():
    xs,ys = row.geometry.coords.xy
    for point1,point2 in pairwise(zip(xs,ys)):
        index1 = all_points_set.index(point1)
        index2 = all_points_set.index(point2)
        adjacency[index1,index2]=True
        adjacency[index2,index1]=True
        
all_points = np.array(all_points_set)
del all_points_set

# Define posterior log likelihood

def get_heights(points):
    return np.array(terrain_raster.interp(x=DataArray(points[:,0]),y=DataArray(points[:,1])))[0]
    
slope_logpdf = norm(scale=options.slope_prior_std).logpdf
offset_logpdf = norm(scale=options.mismatch_prior_std).logpdf
llcount=0
def minus_log_likelihood(point_offsets):
    global llcount
    llcount += 1
    point_offsets = point_offsets.reshape(all_points.shape) # as optimize flattens point_offsets
    points_to_interpolate = all_points + point_offsets
    zs = get_heights(points_to_interpolate)
    distances = distance_matrix(points_to_interpolate,points_to_interpolate) # fixme optimise sparse
    distances += np.eye(num_points) #  can ditch once optimized
    neighbour_slopes = np.arctan(adjacency * abs(zs[:, None] - zs[None, :]) / distances)*180/np.pi # fixme optimize with sparse
    neighbour_likelihood = slope_logpdf(neighbour_slopes).sum() # fixme can optimize if needed by approximating the tan of normal pdf. also, this currently includes non-neighbours as constant
    offset_distances = ((point_offsets**2).sum(axis=1))**0.5 # again, could optimize by approximating square of pdf
    offset_likelihood = offset_logpdf(offset_distances).sum()
    return -(neighbour_likelihood+offset_likelihood)

# Test function

if options.just_lltest:
    offset_unit_vector = np.zeros((num_points,2),float)
    for i in range(num_points):
        offset_unit_vector[i]=np.array([(i//3)%3,i%3])-1
    original_lls = [30839.600582931576,30902.71895798501,31176.373047124474,31298.855321002204,31565.480400726177]
    for i,oll in enumerate(original_lls):
        ll=minus_log_likelihood(offset_unit_vector*i)
        passed=(ll==oll)
        print (f"{i=} {oll=} {ll=} {passed=}")
        
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
    if lldiff<25: #fixme magic number 
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
optimal_zs = get_heights(final_points)
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
