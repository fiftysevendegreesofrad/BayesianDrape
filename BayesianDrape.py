# -*- coding: utf-8 -*-

"""
***************************************************************************
*   Copyright Crispin Cooper 2021                                         *
***************************************************************************
"""

from optparse import OptionParser
import rioxarray # gdal raster is another option should this give trouble though both interpolate with scipy underneath
from xarray import DataArray
from scipy.optimize import minimize,Bounds
from scipy.stats import norm,expon
from scipy.spatial import distance_matrix
import scipy.sparse
from scipy.sparse import lil_matrix,coo_matrix
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
    #options.shapefile = "data/test_awkward_link.shp"
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

# Build point inverse distance matrix which also serves as adjacency matrix

num_points = len(all_points_set)
print (f"{num_points=}")
distances = lil_matrix((num_points,num_points))
adjacency = lil_matrix((num_points,num_points),dtype=bool)
for _,row in net_df.iterrows():
    xs,ys = row.geometry.coords.xy
    for point1,point2 in pairwise(zip(xs,ys)):
        index1 = all_points_set.index(point1)
        index2 = all_points_set.index(point2)
        assert index1!=index2
        (x1,y1),(x2,y2) = point1,point2
        distances[index1,index2]=((x2-x1)**2+(y2-y1)**2)**0.5
        adjacency[index1,index2]=True
        # provided we optimize all points together, we don't store the reverse adjacency 
        # otherwise each gradient likelihood is counted twice, breaking log likelihood
        # if we did want to compute gradient for a single point, we should compute reverse adjacency and also halve all gradient log likelihoods
        # adjacency[index2,index1]=True 
        
all_points = np.array(all_points_set)
del all_points_set
distances = distances.tocsr()
adjacency = adjacency.tocsr()

# Define priors

slope_logpdf = norm(scale=options.slope_prior_std).logpdf
#grade_mean = np.tan(options.slope_prior_std*np.pi/180)
#grade_exp_dist_lambda = 1/grade_mean
#log_grade_exp_dist_lambda = np.log(grade_exp_dist_lambda)
def grade_logpdf(height,dist):
    #return log_grade_exp_dist_lambda - grade_exp_dist_lambda*x
    slope = np.arctan(height,dist)
    return slope_logpdf(slope)

max_coord_displacement=100 # todo take from command line, also interpolation array size
max_displacement = (2**0.5) * max_coord_displacement
dist_range = np.arange(100)/20*max_displacement # 20 was a bug but possibly what made it work as it extends approximation beyond max_coord_displacement
offset_logpdf = norm(scale=options.mismatch_prior_std).logpdf
approx_squareoffset_logpdf = interp1d(dist_range**2,offset_logpdf(dist_range),bounds_error=True,fill_value=-np.inf) # doing this reduced looking up normal pdf from 35% to 10% of runtime of minus_log_likelihood

# wrap interpolators in functions for profiling stats
approximate_offset_prior = False
def approx_squareoffset_logpdff(x):
    if approximate_offset_prior:
        return approx_squareoffset_logpdf(x)
    else:
        return offset_logpdf(x**0.5)
        
def terrain_interpolatorf(x):
    return terrain_interpolator(x)

def sparse_diag(x):
    n, = x.shape
    d = coo_matrix((n,n))
    d.setdiag(x)
    return d.tocsr()

use_dense_matrices = (num_points<200)
if use_dense_matrices: 
    distances = distances.toarray()
    adjacency = adjacency.toarray()

# Define posterior log likelihood

llcount=0

def minus_log_likelihood(point_offsets):
    global llcount
    llcount += 1
    point_offsets = point_offsets.reshape(all_points.shape) # as optimize flattens point_offsets
    points_to_interpolate = all_points + point_offsets
    zs = terrain_interpolatorf(points_to_interpolate)
    
    if use_dense_matrices:
        all_heightdiffs = abs(zs[:, None] - zs[None, :])
        neighbour_heightdiffs = all_heightdiffs[np.nonzero(adjacency)]
        neighbour_distances = distances[np.nonzero(adjacency)]
        neighbour_likelihood = grade_logpdf(neighbour_heightdiffs,neighbour_distances).sum()
    else:
        #the following lines are equivalent to computing the following only for neighbours:
        #neighbour_grades = inverse_distances * abs(zs[:, None] - zs[None, :]) 
        #i.e. computing change in height divided by distance for all neighbours
        #(conveniently, for non-neighbours, inversedistance=1/inf=0 and they don't appear in the sparse matrix so are not computed)
        # .data pulls out all explicit elements including explicit zeros in csr matrix
        sdz = sparse_diag(zs)
        neighbour_grades = abs((inverse_distances * sdz - sdz * inverse_distances).data) # nb * is now matrix mult. is there a name for a*b-b*a ?
        neighbour_likelihood = grade_logpdf(neighbour_grades).sum()

    offset_square_distances = ((point_offsets**2).sum(axis=1))
    offset_likelihood = approx_squareoffset_logpdff(offset_square_distances).sum()
    return -(neighbour_likelihood+offset_likelihood)

# Test function

if options.just_lltest:
    print ("Beginning LL test")
    offset_unit_vector = np.zeros((num_points,2),float)
    
    from timeit import Timer
    ncalls = 1
    nrepeats = 1
    t = Timer(lambda: minus_log_likelihood(offset_unit_vector))
    print("Current impl time:",min(t.repeat(number=ncalls,repeat=nrepeats)))
    # if hasattr(adjacency,"toarray"):
        # nonsparse_adj = adjacency.toarray()
        # t = Timer(lambda: minus_log_likelihood(offset_unit_vector,adjacency=nonsparse_adj))
        # print("Nonsparse time:",min(t.repeat(number=ncalls,repeat=nrepeats)))
    
    
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

def callback(x):
    global llcount,last_ll,result
    ll = -minus_log_likelihood(x)
    lldiff = abs(ll-last_ll)
    last_ll = ll
    print (f"callback {llcount=} {ll=} {lldiff=}")
    llcount=0
    
init_guess = np.zeros((num_points*2),float)
initial_log_likelihood = -minus_log_likelihood(init_guess)
last_ll = initial_log_likelihood
# Bounds are needed to stop the optimizer wandering beyond the furthest approximated distance of the offset prior, which becomes flat at that point
bounds = Bounds(-max_coord_displacement,max_coord_displacement) # not used currently as bounds optimizer not working
print ("Starting optimizer")
result = minimize(minus_log_likelihood,init_guess,callback = callback) 
print (f"Finished optimizing: {result['success']=} {result['message']}")
final_offsets = result["x"].reshape(all_points.shape)
final_points = all_points + final_offsets
optimal_zs = terrain_interpolator(final_points)
final_log_likelihood = -minus_log_likelihood(final_offsets)

# Print result report

offset_distances = ((final_offsets**2).sum(axis=1))**0.5
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
