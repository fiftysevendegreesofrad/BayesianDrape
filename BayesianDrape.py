# -*- coding: utf-8 -*-

"""
***************************************************************************
*   Copyright Crispin Cooper 2021                                         *
***************************************************************************
"""

from optparse import OptionParser
import rioxarray # gdal raster is another option 
from xarray import DataArray
from scipy.optimize import minimize,Bounds
from scipy.stats import norm,expon,describe
from scipy.spatial import distance_matrix
from scipy.special import erfc
import scipy.sparse
from scipy.sparse import lil_matrix,coo_matrix
from torch_interpolations import RegularGridInterpolator
import geopandas as gp
import pandas as pd
import numpy as np
from ordered_set import OrderedSet
from itertools import tee,combinations
from shapely.geometry import LineString
import sys
import torch

using_cuda = False # torch.cuda.is_available() - but doesn't work for me currently as old hardware
if using_cuda:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    def np_to_torch(x):
        return torch.from_numpy(x).cuda()
else:
    def np_to_torch(x):
        return torch.from_numpy(x)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
    
op = OptionParser()
op.add_option("--TERRAIN-INPUT",dest="terrainfile",help="[REQUIRED] Terrain model",metavar="FILE")
op.add_option("--POLYLINE-INPUT",dest="shapefile",help="[REQUIRED] Polyline feature class e.g. network or GPS trace",metavar="FILE")
op.add_option("--OUTPUT",dest="outfile",help="[REQUIRED] Output feature class",metavar="FILE")
op.add_option("--SLOPE-PRIOR-SCALE",dest="slope_prior_std",help="[REQUIRED] Scale of exponential prior for path slope (equivalent to mean slope)",metavar="ANGLE_IN_DEGREES",type="float")
op.add_option("--SPATIAL-MISMATCH-PRIOR-STD",dest="mismatch_prior_std",help="[REQUIRED] Standard deviation of zero-centred Gaussian prior for spatial mismatch (in spatial units of projection)",metavar="DISTANCE",type="float")
op.add_option("--SPATIAL-MISMATCH-MAX",dest="mismatch_max",help="Maximum permissible spatial mismatch (in spatial units of projection; defaults to 4x mismatch prior std)",metavar="DISTANCE",type="float")
op.add_option("--SLOPE-CONTINUITY-PARAM",dest="slope_continuity",help="Slope continuity parameter",metavar="X",type="float",default=0.38)
op.add_option("--JUST-LLTEST",dest="just_lltest",action="store_true",help="Test mode")
op.add_option("--GRADIENT-NEIGHBOUR-DIFFERENCE-OUTPUT",dest="grad_neighbour_diff_file",help="Output for distribution of neighbour gradient differences (for computing autocorrelation)",metavar="FILE")

(options,args) = op.parse_args()

if not options.just_lltest:
    missing_options = []
    for option in op.option_list:
        if option.help.startswith(r'[REQUIRED]') and eval('options.' + option.dest) == None:
            missing_options.extend(option._long_opts)
    if len(missing_options) > 0:
        op.error('Missing REQUIRED parameters: ' + str(missing_options))

if not options.mismatch_max:
    options.mismatch_max = options.mismatch_prior_std * 4

net_df = gp.read_file(options.shapefile)
terrain_raster = rioxarray.open_rasterio(options.terrainfile)

print (f"{net_df.crs=}\nterrain raster crs??")
# todo assert these projections are the same

terrain_xs = np_to_torch(np.array(terrain_raster.x,np.float64))
terrain_ys = np_to_torch(np.flip(np.array(terrain_raster.y,np.float64)).copy() )
terrain_data = np_to_torch(np.flip(terrain_raster.data[0],axis=0).T.copy()) # fixme does this generalize to other projections?

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
# provided we optimize all points together, we only store distances in one direction
# otherwise each gradient likelihood is counted twice, which breaks log likelihood
# if we did ever want to compute log likelihood for a single point, we would need a symmetric distance matrix
num_points = len(all_points_set)
print (f"{num_points=}")
distances = lil_matrix((num_points,num_points))
for _,row in net_df.iterrows():
    xs,ys = row.geometry.coords.xy
    for point1,point2 in pairwise(zip(xs,ys)):
        index1 = all_points_set.index(point1)
        index2 = all_points_set.index(point2)
        assert index1!=index2
        (x1,y1),(x2,y2) = point1,point2
        dist = ((x2-x1)**2+(y2-y1)**2)**0.5
        assert dist>0
        distances[index1,index2]=dist
        
all_points = np.array(all_points_set)
del all_points_set
distances = distances.tocsr()
mean_segment_length = np.mean(distances.data)

print (f"{distances.data.min()=}")

# Compute slope autocorr distribution

if options.grad_neighbour_diff_file:
    neighbour_grade_diffs = []
    one_side_grade = []
    zs = terrain_interpolator(all_points)
    # for each pair of neighbour segments compute difference in slope
    # ...to get neighbour segment pairs, iterate through points, taking combinations for each
    symmetric_distances = distances + distances.T
    for row_idx in range(num_points):
        a,b,d = scipy.sparse.find(symmetric_distances.getrow(row_idx))
        assert np.all(a==0) # redundant index in a single row result
        neighbours = zip(b,d)
        for (idx1,d1),(idx2,d2) in combinations(neighbours,2):
            h1 = zs[idx1]
            h2 = zs[idx2]
            h0 = zs[row_idx]
            g1 = abs(h1-h0)/d1
            g2 = abs(h2-h0)/d2
            neighbour_grade_diffs.append(abs(g1-g2))
            one_side_grade.append(g1)
    pd.DataFrame.from_dict({"grade_diff":neighbour_grade_diffs,"one_side_grade":one_side_grade}).to_csv(options.grad_neighbour_diff_file)
    sys.exit(0)
    
# Define priors - both implemented by hand for speed and compatibility with autodiff

# Exponential grade prior - though we convert to a slope prior for precision reasons
grade_mean = np.tan(options.slope_prior_std*np.pi/180)
print (f"{grade_mean=}")
grade_exp_dist_lambda = 1/grade_mean
def my_exp_logpdf(grade):
    return -np.log(grade_exp_dist_lambda) - grade_exp_dist_lambda*grade

sc = options.slope_continuity * grade_exp_dist_lambda
log_normalizing_constant = -np.log ( ( np.pi**0.5 * np.exp( grade_exp_dist_lambda**2/4/sc ) * erfc (grade_exp_dist_lambda/2/sc**0.5) ) \
                                    / 2 / sc**0.5 \
                                    )
def grade_logpdf(grade):
    return log_normalizing_constant - grade_exp_dist_lambda*grade - sc*grade**2 

# 2d Gaussian offset prior
sigma = options.mismatch_prior_std
sigma_sq = sigma**2
k = -np.log(sigma)-0.5*np.log(2*np.pi)
def squareoffset_logpdf(x):
    return k-(x/sigma_sq)/2 
        
print (f"{mean_segment_length=}")

# test case for gradient priors

show_distributions = False
if show_distributions:
    from matplotlib import pyplot as plt
    even_slope_equal_dists = np.array([1,2,3])
    uneven_slope_equal_dists = np.array([1,1,3])
    equal_dists = np.array([1,1])
    unequal_dists = np.array([0.5,1.5])

    def slope_prior_test(heights,neighbour_distances):
        mean_segment_length = np.mean(neighbour_distances)
        neighbour_heightdiffs = np.diff(heights)
        length_weighted_neighbour_likelihood = grade_logpdf(neighbour_heightdiffs/neighbour_distances)*neighbour_distances / mean_segment_length
        return length_weighted_neighbour_likelihood.sum()
        
    print (f"{slope_prior_test(even_slope_equal_dists,equal_dists)=}")
    print (f"{slope_prior_test(uneven_slope_equal_dists,equal_dists)=}")
    print (f"{slope_prior_test(even_slope_equal_dists,unequal_dists)=}")
    
    grade = np.arange(1000)/1000
    e = my_exp_logpdf(grade)
    me = grade_logpdf(grade)
    plt.plot(grade,e,label="exp")
    plt.plot(grade,me,label="mine")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("logpdf")
    plt.show()
    
    sys.exit(0)

# Define posterior log likelihood

n1s,n2s,neighbour_distances = scipy.sparse.find(distances)
neighbour_weights = neighbour_distances / mean_segment_length
neighbour_inv_distances = np.asarray(distances[n1s,n2s])[0]**-1
del distances
del neighbour_distances

llcount=0

all_points = np_to_torch(all_points)
neighbour_inv_distances = np_to_torch(neighbour_inv_distances)
neighbour_weights = np_to_torch(neighbour_weights)

def minus_log_likelihood(point_offsets):
    if not torch.is_tensor(point_offsets):
        point_offsets = np_to_torch(point_offsets)
    global llcount
    llcount += 1
    point_offsets = torch.reshape(point_offsets,all_points.shape) # as optimize flattens point_offsets
    points_to_interpolate = all_points + point_offsets

    # calling contiguous() silences the performance warning from PyTorch
    # swapping dimensions on our definition of point_offsets, and ensuring all tensors are created contiguous, gives maybe 10% speed boost - reverting for now as that's premature
    zs = terrain_interpolator([points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous()])
    neighbour_heightdiffs = abs(zs[n1s]-zs[n2s]) 
    neighbour_grades = neighbour_heightdiffs*neighbour_inv_distances
    length_weighted_neighbour_likelihood = grade_logpdf(neighbour_grades)*neighbour_weights
    neighbour_likelihood = length_weighted_neighbour_likelihood.sum()
    
    offset_square_distances = ((point_offsets**2).sum(axis=1))
    offset_likelihood = squareoffset_logpdf(offset_square_distances).sum()

    return -(neighbour_likelihood+offset_likelihood)

def minus_log_likelihood_gradient(point_offsets):
    if not torch.is_tensor(point_offsets):
        point_offsets = np_to_torch(point_offsets)
    point_offsets.requires_grad = True
    minus_log_likelihood(point_offsets).backward()
    return point_offsets.grad

# Test function

if options.just_lltest:
    print ("Beginning LL test")
    offset_unit_vector = np.zeros((num_points,2),float)
    grad_test_input = torch.flatten(np_to_torch(offset_unit_vector))
    from timeit import Timer
    ncalls = 100
    nrepeats = 5
    
    if options.shapefile=="data/test_awkward_link.shp":
        original_lls = [579.0005179822035, 593.3853491831525, 629.0507280920274, 688.2552177220412, 783.6203274627381]
        old_gradient = np.array([-0.55813296,-0.12771772])
    elif options.shapefile=="data/biggertest.shp":
        original_lls = [27505.175830755226, 27800.565463084426, 28517.842110572517, 29590.674711176584, 30969.7985891239]
        old_gradient = np.array([-0.09709856, 0.00749611])
    else:
        original_lls = [0,0,0,0,0]
        old_gradient = None
    
    print (f"{minus_log_likelihood(grad_test_input)=}")
    
    print ("Old gradient[0]",old_gradient)
    print("New gradient[0]",minus_log_likelihood_gradient(grad_test_input)[0:2].numpy())
    t = Timer(lambda: minus_log_likelihood_gradient(grad_test_input))
    print("Current gradient time:",min(t.repeat(number=ncalls,repeat=nrepeats)))
    
    for i in range(num_points):
        offset_unit_vector[i]=np.array([(i//3)%3,i%3])-1
    
    new_lls = []
    for i,oll in enumerate(original_lls):
        ll = float(minus_log_likelihood(torch.flatten(np_to_torch(offset_unit_vector*i))))
        new_lls.append(ll)
        passed=(ll==oll)
        print (f"{i=} {oll=} {ll=} {passed=}")
    print (f"{new_lls=}")
        
    sys.exit(0)

# Run optimizer

def callback(x):
    global llcount,last_ll,result
    ll = float(-minus_log_likelihood(x))
    lldiff = abs(ll-last_ll)
    last_ll = ll
    print (f"callback {llcount=} {ll=} {lldiff=}")
    llcount=0
    
init_guess = np_to_torch(np.zeros((num_points*2),float))
initial_log_likelihood = -minus_log_likelihood(init_guess)
last_ll = initial_log_likelihood
bounds = Bounds(-options.mismatch_max,options.mismatch_max) 
print ("Starting optimizer")
# setting maxiter=200 gives nice results but can we do better? fixme
result = minimize(minus_log_likelihood,init_guess,callback = callback,bounds=bounds,jac=minus_log_likelihood_gradient,options=dict(maxiter=200)) 
print (f"Finished optimizing: {result['success']=} {result['message']}")
final_offsets = result["x"].reshape(all_points.shape)
final_points = all_points + final_offsets
optimal_zs = terrain_interpolator([final_points[:,0],final_points[:,1]])
final_log_likelihood = -minus_log_likelihood(final_offsets)

# Print result report

offset_distances = ((final_offsets**2).sum(axis=1))**0.5
mean_offset_dist = offset_distances.mean()
max_offset_dist = offset_distances.max()
print (f"{initial_log_likelihood=}\n{final_log_likelihood=}\n{mean_offset_dist=}\n{max_offset_dist=}")

# Reconstruct geometries
print ("Reconstructing geometries")
xs = all_points[:,0]
ys = all_points[:,1]
net_df.geometry = net_df.apply(lambda row: LineString([(xs[pt_index],ys[pt_index],optimal_zs[pt_index]) for pt_index in row.point_indices]),axis=1)
del net_df["point_indices"]

print (f"Writing output to {options.outfile}") 
net_df.to_file(options.outfile)
