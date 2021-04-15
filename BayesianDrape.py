# -*- coding: utf-8 -*-

"""
***************************************************************************
*   Copyright Crispin Cooper 2021                                         *
***************************************************************************
"""

from scipy.optimize import minimize,Bounds
from scipy.special import erfc
import scipy.sparse
from scipy.sparse import lil_matrix,coo_matrix
from torch_interpolations import RegularGridInterpolator
import numpy as np
from ordered_set import OrderedSet
from itertools import tee,combinations
from shapely.geometry import LineString
import torch
from collections import namedtuple

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

slope_continuity_param_default = 0.38
slope_prior_scale_default = 2.2

def build_model(terrain_index_xs,terrain_index_ys,terrain_zs,
                geometries,
                slope_prior_scale=None,mismatch_prior_std=None,slope_continuity=None,
                simpledraped_geometries_mask=None,decoupled_geometries_mask=None,
                use_cuda=False,
                print_callback=print):

    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        def np_to_torch(x):
            return torch.from_numpy(x).cuda()
    else:
        def np_to_torch(x):
            return torch.from_numpy(x)
            
    if slope_prior_scale==None:
        slope_prior_scale=slope_prior_scale_default
    if slope_continuity==None:
        slope_continuity=slope_continuity_param_default

    # computed parameter defaults
    if simpledraped_geometries_mask is None:
        simpledraped_geometries_mask = [0]*len(geometries)
    if decoupled_geometries_mask is None:
        decoupled_geometries_mask = [0]*len(geometries)
    if mismatch_prior_std is None:
        cellsizex = abs(terrain_index_xs[1]-terrain_index_xs[0])
        cellsizey = abs(terrain_index_ys[1]-terrain_index_ys[0])
        mismatch_prior_std = max(cellsizex,cellsizey)
    
    terrain_xs = np_to_torch(terrain_index_xs.copy())
    terrain_ys = np_to_torch(terrain_index_ys.copy())
    terrain_data = np_to_torch(terrain_zs.copy())
    terrain_interpolator_inner = RegularGridInterpolator((terrain_xs,terrain_ys), terrain_data)
    del terrain_xs, terrain_ys, terrain_data
    
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
    # FIXED points currently reflects simple-draped only though could be expanded to include 3d geometries in future

    all_points_sets = [OrderedSet(),OrderedSet(),OrderedSet()]
    ESTIMATED,FIXED,DECOUPLED = range(len(all_points_sets))

    # Feature and point type model (controls behaviour of optimizer at each point)

    # at feature level, decoupled takes precedence over simpledraped as people are likely to apply simpledrapes to large flat areas and decouples to smaller features like bridges within them
    def get_feature_type(decoupled,simpledraped):
        if decoupled:
            return DECOUPLED
        elif simpledraped:
            return FIXED
        else:
            return ESTIMATED

    # conversely at point level, precedence order for endpoints (the only ones with potential conflict) is simpledraped, estimated, decoupled. We keep these in a dict to keep track of conflicts.
    point_to_type = {} 
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

    # first pass through data to resolve point types
    for geom,decoupled,simpledraped in zip(geometries,decoupled_geometries_mask,simpledraped_geometries_mask):
        xs,ys = geom.coords.xy
        feature_type = get_feature_type(decoupled,simpledraped)
        for x,y in zip(xs,ys):
            set_point_type((x,y),feature_type)
    del decoupled_geometries_mask,simpledraped_geometries_mask
    
    # second pass through data to create point model
    point_indices_all_rows = []
    for geom in geometries:
        assert geom.geom_type=='LineString'
        xs,ys = geom.coords.xy
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
    print_callback (total_num_points,"points in total")
    print_callback (num_estimated_points,"points for estimation")
    print_callback (num_decoupled_points,"points are decoupled")
    print_callback (num_fixed_points,"points are fixed")
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

    # Build all matrices (third pass through data)
    # provided we optimize all points together, we only store distances in one direction
    # otherwise each gradient likelihood is counted twice, which breaks log likelihood
    # if we did ever want to compute log likelihood for a single point, we would need a symmetric distance matrix

    for geom in geometries:
        xs,ys = geom.coords.xy
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
    print_callback("Minimum estimated segment length",all_estimated_segment_lengths.min())
    print_callback("Mean estimated segment length",mean_estimated_segment_length)
    del all_estimated_segment_lengths
        
    # Define priors - both implemented by hand for speed and compatibility with autodiff

    # Exponential grade prior
    grade_scale = np.tan(slope_prior_scale*np.pi/180)
    print_callback(f"Slope prior scale of {slope_prior_scale}\N{DEGREE SIGN} gives grade of {grade_scale:.2f}%")
    grade_exp_dist_lambda = 1/grade_scale
    
    sc = slope_continuity * grade_exp_dist_lambda
    log_normalizing_constant = -np.log ( ( np.pi**0.5 * np.exp( grade_exp_dist_lambda**2/4/sc ) * erfc (grade_exp_dist_lambda/2/sc**0.5) ) \
                                        / 2 / sc**0.5 \
                                        )
    def grade_logpdf(grade):
        return log_normalizing_constant - grade_exp_dist_lambda*grade - sc*grade**2 

    # 2d Gaussian offset prior
    sigma = mismatch_prior_std
    sigma_sq = sigma**2
    k = -np.log(sigma)-0.5*np.log(2*np.pi)
    def squareoffset_logpdf(x):
        return k-(x/sigma_sq)/2 
            
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

    def minus_log_likelihood(point_offsets):
        if not torch.is_tensor(point_offsets):
            point_offsets = np_to_torch(point_offsets)
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
        
    def reconstruct_geometries(opt_results):
        final_offsets = opt_results.reshape(all_points_arrays[ESTIMATED].shape)
        final_points = all_points_arrays[ESTIMATED] + final_offsets
        optimal_zs = terrain_interpolator(final_points)
        final_log_likelihood = -minus_log_likelihood(final_offsets)

        # Print report on offsets
        
        offset_distances = ((final_offsets**2).sum(axis=1))**0.5
        mean_offset_dist = offset_distances.mean()
        max_offset_dist = offset_distances.max()
        print_callback (f"{mean_offset_dist=}\n{max_offset_dist=}")

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
        
        return [LineString([get_point_output(pt_index,decoupled_zs) for pt_index in row_point_indices]) for row_point_indices in point_indices_all_rows]
    
    return_dict = dict(grade_logpdf=grade_logpdf,
                  squareoffset_logpdf=squareoffset_logpdf,
                  minus_log_likelihood=lambda x: float(minus_log_likelihood(x)),
                  minus_log_likelihood_gradient=minus_log_likelihood_gradient,
                  initial_guess=np_to_torch(np.zeros((num_estimated_points*2),float)),
                  reconstruct_geometries_from_optimizer_results=reconstruct_geometries,
                  mismatch_prior_std=mismatch_prior_std
                  )
    BayesianDrapeModel = namedtuple("BayesianDrapeModel",return_dict)
    return BayesianDrapeModel(**return_dict)
    
def fit_model(model,max_offset_dist=np.inf,print_callback=print):
    initial_log_likelihood = -model.minus_log_likelihood(model.initial_guess)
    last_ll = initial_log_likelihood

    def callback(x):
        nonlocal last_ll
        ll = float(-model.minus_log_likelihood(x))
        lldiff = abs(ll-last_ll)
        last_ll = ll
        print (f"callback {ll=} {lldiff=}")
    
    bounds = Bounds(-max_offset_dist,max_offset_dist) 
    print_callback ("Starting optimizer")
    # setting maxiter=200 gives nice results but can we do better? fixme
    result = minimize(model.minus_log_likelihood,model.initial_guess,callback = callback,bounds=bounds,jac=model.minus_log_likelihood_gradient,options=dict(maxiter=200)) 
    print_callback (f"Finished optimizing: {result['success']=} {result['message']}")
    
    optimizer_results = result["x"]
    final_log_likelihood = -model.minus_log_likelihood(optimizer_results)
    print_callback (f"{initial_log_likelihood=}\n{final_log_likelihood=}\n")
    
    print_callback ("Reconstructing geometries")
    return [LineString(geom) for geom in model.reconstruct_geometries_from_optimizer_results(optimizer_results)]

def fit_model_from_command_line_options():
    from optparse import OptionParser
    import rioxarray # gdal raster is another option 
    import pandas as pd
    import geopandas as gp
    
    op = OptionParser()
    op.add_option("--TERRAIN-INPUT",dest="terrainfile",help="[REQUIRED] Terrain model",metavar="FILE")
    op.add_option("--POLYLINE-INPUT",dest="shapefile",help="[REQUIRED] Polyline feature class e.g. network or GPS trace",metavar="FILE")
    op.add_option("--OUTPUT",dest="outfile",help="[REQUIRED] Output feature class",metavar="FILE")
    op.add_option("--SLOPE-PRIOR-SCALE",dest="slope_prior_scale",help=f"Scale of exponential prior for path slope (equivalent to mean slope; defaults to {slope_prior_scale_default})",metavar="ANGLE_IN_DEGREES",type="float")
    op.add_option("--SPATIAL-MISMATCH-PRIOR-STD",dest="mismatch_prior_std",help="Standard deviation of zero-centred Gaussian prior for spatial mismatch (in spatial units of projection; defaults to half terrain raster cell size)",metavar="DISTANCE",type="float")
    op.add_option("--SPATIAL-MISMATCH-MAX",dest="mismatch_max",help="Maximum permissible spatial mismatch (in spatial units of projection; defaults to 4x mismatch prior std)",metavar="DISTANCE",type="float")
    op.add_option("--SLOPE-CONTINUITY-PARAM",dest="slope_continuity",help=f"Slope continuity parameter (defaults to {slope_continuity_param_default})",metavar="X",type="float",default=slope_continuity_param_default)
    op.add_option("--GPU",dest="cuda",action="store_true",help="Enable GPU acceleration")
    op.add_option("--SIMPLE-DRAPE-FIELD",dest="simpledrapefield",help="Instead of estimating heights, perform ordinary drape of features over terrain where FIELDNAME=true",metavar="FIELDNAME")
    op.add_option("--DECOUPLE-FIELD",dest="decouplefield",help="Instead of estimating heights, decouple features from terrain where FIELDNAME=true (useful for bridges/tunnels)",metavar="FIELDNAME")
    (options,args) = op.parse_args()
    
    if options.cuda and not torch.cuda.is_available():
            op.error("PyTorch CUDA is not available; try running without --GPU")
            
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
    
    terrain_xs = (np.array(terrain_raster.x,np.float64))
    terrain_ys = (np.flip(np.array(terrain_raster.y,np.float64)) )
    terrain_data = (np.flip(terrain_raster.data[0],axis=0).T) # fixme does this generalize to other projections?
    
    if options.simpledrapefield and options.decouplefield and np.logical_and(net_df[options.simpledrapefield],net_df[options.decouplefield]).any():
        print ("Input contains rows which are both fixed/simple-draped and decoupled - decoupled takes precedence")
        
    if options.simpledrapefield:
        simpledrape_geometries_mask = net_df[options.simpledrapefield]
    else:
        simpledrape_geometries_mask = None
    if options.decouplefield:
        decouple_geometries_mask = net_df[options.decouplefield]
    else:
        decouple_geometries_mask = None
        
    # Build model
        
    model = build_model(terrain_xs,terrain_ys,terrain_data,net_df.geometry,
                        slope_prior_scale = options.slope_prior_scale,
                        mismatch_prior_std = options.mismatch_prior_std,
                        slope_continuity = options.slope_continuity,
                        simpledraped_geometries_mask = simpledrape_geometries_mask,
                        decoupled_geometries_mask = decouple_geometries_mask,
                        use_cuda = options.cuda)
    
    del terrain_xs,terrain_ys,terrain_data,terrain_raster
    
    if not options.mismatch_max:
        options.mismatch_max = model.mismatch_prior_std * 4
    
    net_df.geometry = fit_model(model,options.mismatch_max)

    print (f"Writing output to {options.outfile}") 
    net_df.to_file(options.outfile)

if __name__ == "__main__":
    fit_model_from_command_line_options()
    