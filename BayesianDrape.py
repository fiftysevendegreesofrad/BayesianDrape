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
from scipy.stats import norm,expon,describe
from scipy.spatial import distance_matrix
from scipy.special import erfc
import scipy.sparse
from scipy.sparse import lil_matrix,coo_matrix
from scipy.interpolate import RegularGridInterpolator,interp1d
import geopandas as gp
import pandas as pd
import numpy as np
from ordered_set import OrderedSet
from itertools import tee,combinations
from shapely.geometry import LineString
import sys
from math import fsum

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

# Build point inverse distance matrix and adjacency matrix

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
        # provided we optimize all points together, we don't store the reverse adjacency 
        # otherwise each gradient likelihood is counted twice, breaking log likelihood
        # if we did want to compute gradient for a single point, we should compute reverse adjacency and also halve all gradient log likelihoods
        # adjacency[index2,index1]=True 
        
all_points = np.array(all_points_set)
del all_points_set
distances = distances.tocsr()
mean_segment_length = np.mean(distances.data)

print (f"{distances.data.min()=}")

# Compute slope autocorr distribution
# fixme if std not provided for autocorr, do this efficiently and estimate distribution as appropriate

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
    
# Define priors

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

print (f"{my_exp_logpdf(0)=}")
print (f"{grade_exp_dist_lambda=}")
print (f"{grade_logpdf(0)=}")

slope_angle_range = np.arange(901)/10 # fixme could be more canny with spacing to speed up if needed
slope_pdf_interp_points = grade_logpdf(np.tan(slope_angle_range/180*np.pi))
slope_pdf_interp_points[-1] = slope_pdf_interp_points[-2]*2 # fill last value to prevent it dominating likelihood computations
# fixme can i approximate the tan without precision issues? yes quite possibly, from gradient, as that happened before. only if profiler indicates an issue though and it makes problems outside last interp point
approx_slope_logpdf = interp1d(slope_angle_range,slope_pdf_interp_points,bounds_error=True) # fixme get rid of tan if we keep this also incorporate 'weird'

#slope_prior = "approx_grade_expon"
slope_prior = "exact_grade_expon"
def grade_logpdf_f(grade):
    if slope_prior == "approx_grade_expon":
        return approx_grade_logpdf(grade)
    if slope_prior == "exact_grade_expon":
        return grade_logpdf(grade)
    assert False

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

inverse_distances = distances.copy()
inverse_distances.data**=-1

use_dense_matrices = (num_points<200)
if use_dense_matrices: 
    adjacency = np.zeros((num_points,num_points),bool)
    adjacency[distances.nonzero()] = 1
    distances = distances.toarray()
    
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
        return fsum(length_weighted_neighbour_likelihood)
        
    print (f"{slope_prior_test(even_slope_equal_dists,equal_dists)=}")
    print (f"{slope_prior_test(uneven_slope_equal_dists,equal_dists)=}")
    print (f"{slope_prior_test(even_slope_equal_dists,unequal_dists)=}")
    
    grade = np.arange(1000)/1000
    e = my_exp_logpdf(grade)
    me = grade_logpdf_f(grade)
    plt.plot(grade,e,label="exp")
    plt.plot(grade,me,label="mine")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("logpdf")
    plt.show()
    
    sys.exit(0)

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
        neighbour_distances = distances[np.nonzero(adjacency)] # not efficient
        neighbour_grades = neighbour_heightdiffs/neighbour_distances
    else:
        n1s,n2s,neighbour_distances = scipy.sparse.find(distances) 
        neighbour_inv_distances = np.asarray(inverse_distances[n1s,n2s])[0]
        neighbour_heightdiffs = abs(zs[n1s]-zs[n2s]) 
        neighbour_grades = neighbour_heightdiffs*neighbour_inv_distances

    length_weighted_neighbour_likelihood = grade_logpdf_f(neighbour_grades)*neighbour_distances / mean_segment_length
    neighbour_likelihood = fsum(length_weighted_neighbour_likelihood)
    
    offset_square_distances = ((point_offsets**2).sum(axis=1))
    offset_likelihood = fsum(approx_squareoffset_logpdff(offset_square_distances))
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
    original_lls = [579.0005179822034, 593.3853491831526, 629.0507280920274, 688.2552177220412, 783.620327462738]
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
result = minimize(minus_log_likelihood,init_guess,callback = callback,bounds=bounds,options=dict(maxfun=100000,maxiter=np.inf)) 
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
