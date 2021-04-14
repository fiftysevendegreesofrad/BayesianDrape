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

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

slope_continuity_param_default = 0.38    
op = OptionParser()
op.add_option("--TERRAIN-INPUT",dest="terrainfile",help="[REQUIRED] Terrain model",metavar="FILE")
op.add_option("--POLYLINE-INPUT",dest="shapefile",help="[REQUIRED] Polyline feature class e.g. network or GPS trace",metavar="FILE")
op.add_option("--OUTPUT",dest="outfile",help="[REQUIRED] Output feature class",metavar="FILE")
op.add_option("--SLOPE-PRIOR-SCALE",dest="slope_prior_scale",help="[REQUIRED] Scale of exponential prior for path slope (equivalent to mean slope)",metavar="ANGLE_IN_DEGREES",type="float")
op.add_option("--SPATIAL-MISMATCH-PRIOR-STD",dest="mismatch_prior_std",help="[REQUIRED] Standard deviation of zero-centred Gaussian prior for spatial mismatch (in spatial units of projection)",metavar="DISTANCE",type="float")
op.add_option("--SPATIAL-MISMATCH-MAX",dest="mismatch_max",help="Maximum permissible spatial mismatch (in spatial units of projection; defaults to 4x mismatch prior std)",metavar="DISTANCE",type="float")
op.add_option("--SLOPE-CONTINUITY-PARAM",dest="slope_continuity",help=f"Slope continuity parameter (defaults to {slope_continuity_param_default})",metavar="X",type="float",default=slope_continuity_param_default)
op.add_option("--JUST-LLTEST",dest="just_lltest",action="store_true",help="Test mode")
op.add_option("--GPU",dest="cuda",action="store_true",help="Enable GPU acceleration")
op.add_option("--FIX-FIELD",dest="fixfield",help="Instead of estimating heights, perform ordinary drape of features over terrain where FIELDNAME=true",metavar="FIELDNAME")
op.add_option("--DECOUPLE-FIELD",dest="decouplefield",help="Instead of estimating heights, decouple features from terrain where FIELDNAME=true (useful for bridges/tunnels)",metavar="FIELDNAME")

(options,args) = op.parse_args()

if options.cuda:
    if not torch.cuda.is_available():
        op.error("PyTorch CUDA is not available; try running without --GPU")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    def np_to_torch(x):
        return torch.from_numpy(x).cuda()
else:
    def np_to_torch(x):
        return torch.from_numpy(x)

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

terrain_interpolator_inner = RegularGridInterpolator((terrain_xs,terrain_ys), terrain_data)
del terrain_xs, terrain_ys, terrain_data, terrain_raster
def terrain_interpolator(points_to_interpolate):
    # calling contiguous() silences the performance warning from PyTorch
    # swapping dimensions on our definition of points throughout, and ensuring all tensors are created contiguous, gives maybe 10% speed boost - reverting for now as that's premature
    if not torch.is_tensor(points_to_interpolate):
        points_to_interpolate = np_to_torch(points_to_interpolate)
    return terrain_interpolator_inner([points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous()])

# Todo: add points to linestrings where they cross raster cells or at least every cell-distance
# (latter may make more sense as more expensive to compute gradients at cell boundaries)

# Build point model: we represent each linestring with a point index list 
# point (x,y)s are initially stored in OrderedSets so we can give the same index to matching line endpoints; these OrderedSets are later converted to arrays
# We define them now so we can simultaneously define an enum for the corresponding point type that belongs in each

all_points_sets = [OrderedSet(),OrderedSet(),OrderedSet()]
ESTIMATED,FIXED,DECOUPLED = range(len(all_points_sets))

# Feature and point type model (controls behaviour of optimizer at each point)

point_to_type = {} 

# at feature level, decoupled takes precedence over fixed as people are likely to apply fixes to large flat areas and decouples to smaller features like bridges within them
both_fixed_decoupled_warning_issued = False
def get_feature_type(row):
    global both_fixed_decoupled_warning_issued
    fixed = options.fixfield and row[options.fixfield]
    decoupled = options.decouplefield and row[options.decouplefield]
    if not both_fixed_decoupled_warning_issued and fixed and decoupled:
        print (f"Input contains rows which are both fixed and decoupled - decoupled takes precedence")
        both_fixed_decoupled_warning_issued = True
    if decoupled:
        return DECOUPLED
    elif fixed:
        return FIXED
    else:
        return ESTIMATED

# conversely at point level, precedence order for endpoints (the only ones with potential conflict) is fixed, estimated, decoupled
def set_point_type(coords,feature_type):
    if feature_type == FIXED:
        point_to_type[coords] = FIXED
    elif feature_type == ESTIMATED:
        if coords not in point_to_type or point_to_type[coords] != FIXED:
            point_to_type[coords] = ESTIMATED
    elif feature_type == DECOUPLED:
        if coords not in point_to_type:
            point_to_type[coords] = DECOUPLED
    else: assert False    

# first pass through df to resolve point types
for _,row in net_df.iterrows():
    xs,ys = row.geometry.coords.xy
    feature_type = get_feature_type(row)
    for x,y in zip(xs,ys):
        set_point_type((x,y),feature_type)

# second pass through df to create point model
point_indices_all_rows = []
for _,row in net_df.iterrows():
    assert row.geometry.geom_type=='LineString'
    xs,ys = row.geometry.coords.xy
    point_indices = []
    for x,y in zip(xs,ys):
        point_type = point_to_type[(x,y)]
        index = all_points_sets[point_type].add((x,y))
        point_indices.append((point_type,index))
    point_indices_all_rows.append(point_indices)

num_estimated_points = len(all_points_sets[ESTIMATED])
num_decoupled_points = len(all_points_sets[DECOUPLED])
num_fixed_points = len(all_points_sets[FIXED])
total_num_points = num_estimated_points+num_decoupled_points+num_fixed_points
print (total_num_points,"points in total")
print (num_estimated_points,"points for estimation")
print (num_decoupled_points,"points are decoupled")
print (num_fixed_points,"points are fixed")
assert num_estimated_points > 0

def get_point_type_and_index(point):
    t = point_to_type[point]
    return t,all_points_sets[t].index(point)

# Initialize decoupled_graph which stores distances between all decoupled points and their neighbours (of any type), with its own indexing scheme

decoupled_graph = lil_matrix((total_num_points,total_num_points))
decoupled_group_boundary_points = []

def decoupled_graph_index(pt_type,index):
    # decoupled_graph_nodes_only and other_graph_nodes_only below, rely on this ordering
    if pt_type==DECOUPLED:
        return index
    elif pt_type==ESTIMATED:
        return num_decoupled_points+index
    elif pt_type==FIXED:
        return num_decoupled_points+num_estimated_points+index
    else: assert False

def decoupled_graph_nodes_only(distances):
    return distances[0:num_decoupled_points]
    
def other_graph_nodes_only(distances):
    return distances[num_decoupled_points:]

def decoupled_graph_add(decoupled_index,other_type,other_index,dist):
    i1 = decoupled_graph_index(DECOUPLED,decoupled_index)
    i2 = decoupled_graph_index(other_type,other_index)
    # directed graph - boundary points can flow inwards only
    decoupled_graph[i2,i1] = dist
    if other_type == DECOUPLED:
        decoupled_graph[i1,i2] = dist
    else:
        decoupled_group_boundary_points.append((other_type,other_index))

# Initialize point inverse distance matrices, which also serve as adjacency matrices

distance_matrix_estimated_to_estimated = lil_matrix((num_estimated_points,num_estimated_points))
distance_matrix_estimated_to_fixed = lil_matrix((num_estimated_points,num_fixed_points))

# Build all matrices (third pass through df)
# provided we optimize all points together, we only store distances in one direction
# otherwise each gradient likelihood is counted twice, which breaks log likelihood
# if we did ever want to compute log likelihood for a single point, we would need a symmetric distance matrix

for _,row in net_df.iterrows():
    xs,ys = row.geometry.coords.xy
    for point1,point2 in pairwise(zip(xs,ys)):
        type1,index1 = get_point_type_and_index(point1)
        type2,index2 = get_point_type_and_index(point2)
        assert (type1,index1)!=(type2,index2)
        (x1,y1),(x2,y2) = point1,point2
        dist = ((x2-x1)**2+(y2-y1)**2)**0.5
        assert dist>0
        if (type1,type2)==(ESTIMATED,ESTIMATED):
            distance_matrix_estimated_to_estimated[index1,index2]=dist
        elif (type1,type2)==(ESTIMATED,FIXED):
            distance_matrix_estimated_to_fixed[index1,index2]=dist
        elif (type1,type2)==(FIXED,ESTIMATED):
            distance_matrix_estimated_to_fixed[index2,index1]=dist
        elif type1==DECOUPLED:
            decoupled_graph_add(index1,type2,index2,dist)
        elif type2==DECOUPLED:
            assert type1!=DECOUPLED # handled above
            decoupled_graph_add(index2,type1,index1,dist)
        else: 
            assert (type1,type2)==(FIXED,FIXED) # no need to store

all_points_arrays = [np.array(s) for s in all_points_sets]
del all_points_sets,point_to_type
distance_matrix_estimated_to_estimated = distance_matrix_estimated_to_estimated.tocsr()
distance_matrix_estimated_to_fixed = distance_matrix_estimated_to_fixed.tocsr()
decoupled_graph = decoupled_graph.tocsr()

all_estimated_segment_lengths = np.concatenate((distance_matrix_estimated_to_estimated.data,distance_matrix_estimated_to_fixed.data))
mean_estimated_segment_length = all_estimated_segment_lengths.mean()
print("Minimum estimated segment length",all_estimated_segment_lengths.min())
print("Mean estimated segment length",mean_estimated_segment_length)
del all_estimated_segment_lengths
    
# Define priors - both implemented by hand for speed and compatibility with autodiff

# Exponential grade prior
grade_scale = np.tan(options.slope_prior_scale*np.pi/180)
print (f"Slope prior scale of {options.slope_prior_scale}\N{DEGREE SIGN} gives grade of {grade_scale:.2f}%")
grade_exp_dist_lambda = 1/grade_scale
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
        
# Test case for gradient priors

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
        length_weighted_neighbour_likelihood = grade_logpdf(neighbour_heightdiffs/neighbour_distances)*neighbour_distances / mean_estimated_segment_length
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

# Prepare sparse arrays estimated-to-estimated: two indices into estimated points, weights, inverse distances
sparse_est_est_neighbour1s,sparse_est_est_neighbour2s,sparse_est_est_neighbour_distances = scipy.sparse.find(distance_matrix_estimated_to_estimated)
sparse_est_est_neighbour_weights = sparse_est_est_neighbour_distances / mean_estimated_segment_length
sparse_est_est_neighbour_inv_distances = np.asarray(distance_matrix_estimated_to_estimated[sparse_est_est_neighbour1s,sparse_est_est_neighbour2s])[0]**-1
del sparse_est_est_neighbour_distances
del distance_matrix_estimated_to_estimated

# Prepare sparse arrays estimated-to-fixed: indices into fixed and estimated points, weights, inverse distances
sparse_est_fix_neighbour1s,sparse_est_fix_neighbour2s,sparse_est_fix_neighbour_distances = scipy.sparse.find(distance_matrix_estimated_to_fixed)
sparse_est_fix_neighbour_weights = sparse_est_fix_neighbour_distances / mean_estimated_segment_length
if len(sparse_est_fix_neighbour_distances):
    sparse_est_fix_neighbour_inv_distances = np.asarray(distance_matrix_estimated_to_fixed[sparse_est_fix_neighbour1s,sparse_est_fix_neighbour2s])[0]**-1
else:
    sparse_est_fix_neighbour_inv_distances = np.zeros(0) # numpy zero length arrays must be 0-dimensional, which is inconvenient
del sparse_est_fix_neighbour_distances
del distance_matrix_estimated_to_fixed
if len(all_points_arrays[FIXED]):
    fixed_zs = terrain_interpolator(all_points_arrays[FIXED])
else:
    fixed_zs = np_to_torch(np.zeros(0))

# Convert everything minus_log_likelihood needs to PyTorch
all_points_arrays = [np_to_torch(a) for a in all_points_arrays]
sparse_est_est_neighbour_weights = np_to_torch(sparse_est_est_neighbour_weights)
sparse_est_est_neighbour_inv_distances = np_to_torch(sparse_est_est_neighbour_inv_distances)
sparse_est_fix_neighbour_weights = np_to_torch(sparse_est_fix_neighbour_weights)
sparse_est_fix_neighbour_inv_distances = np_to_torch(sparse_est_fix_neighbour_inv_distances)

llcount=0
def minus_log_likelihood(point_offsets):
    if not torch.is_tensor(point_offsets):
        point_offsets = np_to_torch(point_offsets)
    global llcount
    llcount += 1
    point_offsets = torch.reshape(point_offsets,all_points_arrays[ESTIMATED].shape) # as optimize flattens point_offsets
    
    # Log likelihood of grades between all neighbouring pairs of estimated points
    estimated_zs = terrain_interpolator(all_points_arrays[ESTIMATED] + point_offsets)
    est_est_neighbour_heightdiffs = abs(estimated_zs[sparse_est_est_neighbour1s]-estimated_zs[sparse_est_est_neighbour2s]) 
    est_est_neighbour_grades = est_est_neighbour_heightdiffs*sparse_est_est_neighbour_inv_distances
    est_est_neighbour_likelihood = (grade_logpdf(est_est_neighbour_grades)*sparse_est_est_neighbour_weights).sum()
    
    # Log likelihood of grades between all estimated-fixed point pairs
    est_fix_neighbour_heightdiffs = abs(estimated_zs[sparse_est_fix_neighbour1s]-fixed_zs[sparse_est_fix_neighbour2s]) 
    est_fix_neighbour_grades = est_fix_neighbour_heightdiffs*sparse_est_fix_neighbour_inv_distances
    est_fix_neighbour_likelihood = (grade_logpdf(est_fix_neighbour_grades)*sparse_est_fix_neighbour_weights).sum()
    
    # Log likelihood of point offsets
    offset_square_distances = ((point_offsets**2).sum(axis=1))
    offset_likelihood = squareoffset_logpdf(offset_square_distances).sum()

    return -(est_est_neighbour_likelihood + est_fix_neighbour_likelihood + offset_likelihood)

def minus_log_likelihood_gradient(point_offsets):
    if not torch.is_tensor(point_offsets):
        point_offsets = np_to_torch(point_offsets)
    point_offsets.requires_grad = True
    minus_log_likelihood(point_offsets).backward()
    return point_offsets.grad

# Test function

if options.just_lltest:
    print ("Beginning LL test")
    offset_unit_vector = np.zeros((num_estimated_points,2),float)
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
    
    for i in range(num_estimated_points):
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
    
init_guess = np_to_torch(np.zeros((num_estimated_points*2),float))
initial_log_likelihood = -minus_log_likelihood(init_guess)
last_ll = initial_log_likelihood
bounds = Bounds(-options.mismatch_max,options.mismatch_max) 
print ("Starting optimizer")
# setting maxiter=200 gives nice results but can we do better? fixme
result = minimize(minus_log_likelihood,init_guess,callback = callback,bounds=bounds,jac=minus_log_likelihood_gradient,options=dict(maxiter=200)) 
print (f"Finished optimizing: {result['success']=} {result['message']}")
final_offsets = result["x"].reshape(all_points_arrays[ESTIMATED].shape)
final_points = all_points_arrays[ESTIMATED] + final_offsets
optimal_zs = terrain_interpolator(final_points)
final_log_likelihood = -minus_log_likelihood(final_offsets)

# Print result report

offset_distances = ((final_offsets**2).sum(axis=1))**0.5
mean_offset_dist = offset_distances.mean()
max_offset_dist = offset_distances.max()
print (f"{initial_log_likelihood=}\n{final_log_likelihood=}\n{mean_offset_dist=}\n{max_offset_dist=}")

# Interpolate decoupled points: iterate through decoupled_group_boundary_points, as they are likely fewer in number than the decoupled points themselves

def get_point_output(index_tuple,decoupled_zs):
    pt_type,pt_index = index_tuple
    x,y = all_points_arrays[pt_type][pt_index]
    if pt_type==ESTIMATED:
        return x,y,float(optimal_zs[pt_index])
    elif pt_type==FIXED:
        return x,y,float(fixed_zs[pt_index])
    elif pt_type==DECOUPLED:
        return x,y,decoupled_zs[pt_index]
    else: assert False

decoupled_weighted_z_sum = np.zeros(num_decoupled_points)
decoupled_weight_sum = np.zeros(num_decoupled_points)

if decoupled_group_boundary_points:
    print ("Interpolating decoupled points")
for pt in decoupled_group_boundary_points:
    pt_type,pt_index = pt
    _,_,boundary_pt_z = get_point_output(pt,None)
    ind = decoupled_graph_index(pt_type,pt_index)
    assert ind >= num_decoupled_points
    # n.b. if performance is an issue the method below can take multiple indices at once
    distances = scipy.sparse.csgraph.dijkstra(decoupled_graph,directed=True,indices=ind)
    
    # sanity check: the only node with zero distance is the current boundary node
    (zerodist_node,), = np.where(distances==0)
    assert zerodist_node==ind 
    
    # sanity check: all other non-decoupled graph nodes should have distance==inf
    ogn = other_graph_nodes_only(distances)
    assert ogn.shape==(num_estimated_points+num_fixed_points,) 
    ogn_check = np.zeros(ogn.shape)+np.inf
    ogn_check[ind-num_decoupled_points]=0
    assert np.all(ogn==ogn_check)
    
    weights = decoupled_graph_nodes_only(distances)**-1
    decoupled_weighted_z_sum += weights * boundary_pt_z
    decoupled_weight_sum += weights
    
decoupled_zs = decoupled_weighted_z_sum/decoupled_weight_sum

# Reconstruct geometries

print ("Reconstructing geometries")
net_df.geometry = [LineString([get_point_output(pt_index,decoupled_zs) for pt_index in row_point_indices]) for row_point_indices in point_indices_all_rows]

print (f"Writing output to {options.outfile}") 
net_df.to_file(options.outfile)
