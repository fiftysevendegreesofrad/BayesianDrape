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
from collections import namedtuple,defaultdict

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def grade_change(g1,g2):
    '''Rotating grades g1 and g2 such that g1 is horizontal, what is abs(g2)?'''
    # answer derived from triangle rule a^2==b^2+c^2-2bc cos A
    ratio = (-g1*g2-1)/((1+g1**2)*(1+g2**2))**0.5
    # to get angle we would arccos(ratio), but we want ...
    return abs(((1-ratio**2)**0.5)/ratio)

slope_continuity_scale_default = 0.42
slope_prior_scale_default = 2.2
curvature_scale_default = 0.0036
curvature_shape_default = 3.20

def build_model(terrain_index_xs,terrain_index_ys,terrain_zs,
                geometries,
                slope_prior_scale=None,mismatch_prior_std=None,slope_continuity_scale=None,slope_continuity_desired_impact=None,
                use_curvature_prior=False,curvature_scale=None,curvature_shape = None,
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
    assert not (slope_continuity_scale is not None and slope_continuity_desired_impact is not None)
    if slope_continuity_scale==None and slope_continuity_desired_impact==None:
        slope_continuity_scale=slope_continuity_scale_default

    if curvature_scale == None:
        curvature_scale = curvature_scale_default
    if curvature_shape == None:
        curvature_shape = curvature_shape_default

    # computed parameter defaults
    if simpledraped_geometries_mask is None:
        simpledraped_geometries_mask = [0]*len(geometries)
    if decoupled_geometries_mask is None:
        decoupled_geometries_mask = [0]*len(geometries)
    if mismatch_prior_std is None:
        cellsizex = abs(terrain_index_xs[1]-terrain_index_xs[0])
        cellsizey = abs(terrain_index_ys[1]-terrain_index_ys[0])
        mismatch_prior_std = max(cellsizex,cellsizey)/2
    
    terrain_xs = np_to_torch(terrain_index_xs.copy())
    terrain_ys = np_to_torch(terrain_index_ys.copy())
    terrain_data = np_to_torch(terrain_zs.copy())
    terrain_min_height = float(terrain_data.min())
    terrain_max_height = float(terrain_data.max())
    print_callback(f"Terrain height range from {terrain_min_height:.2f} to {terrain_max_height:.2f}")
    terrain_interpolator_not_pytorch = RegularGridInterpolator((terrain_xs,terrain_ys), terrain_data)
    del terrain_xs, terrain_ys, terrain_data
    
    def terrain_interpolator(points_to_interpolate):
        # calling contiguous() silences the performance warning from PyTorch
        # swapping dimensions on our definition of points throughout, and ensuring all tensors are created contiguous, gives maybe 10% speed boost - reverting for now as that's premature
        if not torch.is_tensor(points_to_interpolate):
            points_to_interpolate = np_to_torch(points_to_interpolate)
        return terrain_interpolator_not_pytorch([points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous()])

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

    # Storage for all gradient tests we will conduct - a bit like a sparse adjacency matrix but with types+indices
    gradient_test_p1_types = []
    gradient_test_p1_indices = []
    gradient_test_p2_types = []
    gradient_test_p2_indices = []
    gradient_test_distances = []
    def add_gradient_test(type1,index1,type2,index2,dist):
        assert index1<len(all_points_sets[type1])
        assert index2<len(all_points_sets[type2])
        assert type(dist)==float
        gradient_test_p1_types.append(type1)
        gradient_test_p1_indices.append(index1)
        gradient_test_p2_types.append(type2)
        gradient_test_p2_indices.append(index2)
        gradient_test_distances.append(dist)
        
    def all_gradient_tests():
        zipped = zip(gradient_test_p1_types,gradient_test_p1_indices,gradient_test_p2_types,gradient_test_p2_indices)
        for i,(t1,i1,t2,i2) in enumerate(zipped):
            yield i,(t1,i1),(t2,i2)
    
    def all_neighbouring_points():
        for geom in geometries:
            xs,ys = geom.coords.xy
            for point1,point2 in pairwise(zip(xs,ys)):
                type1,index1 = get_point_type_and_index(point1)
                type2,index2 = get_point_type_and_index(point2)
                assert (type1,index1)!=(type2,index2)
                yield (type1,index1,point1),(type2,index2,point2)
    
    # pass through data again to determine which fixed points are next to others for curvature test
    fixed_point_adjacent_to_nonfixed = set()
    if use_curvature_prior:
        for (type1,index1,_),(type2,index2,_) in all_neighbouring_points():
            if type1==FIXED and type2!=FIXED:
                fixed_point_adjacent_to_nonfixed.add(index1)
            if type2==FIXED and type1!=FIXED:
                fixed_point_adjacent_to_nonfixed.add(index2)
    
    # Build all matrices (another pass through data)
    # provided we optimize all points together, we only store distances in one direction
    # otherwise each gradient likelihood is counted twice, which breaks log likelihood
    # if we did ever want to compute log likelihood for a single point, we would need a symmetric distance matrix

    for (type1,index1,(x1,y1)),(type2,index2,(x2,y2)) in all_neighbouring_points():
        dist = ((x2-x1)**2+(y2-y1)**2)**0.5
        assert dist>0
        
        if ((type1,type2)!=(FIXED,FIXED)
            or ((type1,type2)==(FIXED,FIXED) and curvature_test and (index1 in fixed_point_adjacent_to_nonfixed or index2 in fixed_point_adjacent_to_nonfixed))):
                add_gradient_test(type1,index1,type2,index2,dist)
            
        if type1==DECOUPLED:
            decoupled_graph_add(index1,type2,index2,dist)
        elif type2==DECOUPLED:
            assert type1!=DECOUPLED # handled above
            decoupled_graph_add(index2,type1,index1,dist)
    
    del fixed_point_adjacent_to_nonfixed
    
    # build curvature test
    curvature_test_g1_indices = []
    curvature_test_g2_indices = []
    curvature_test_g1_senses = []
    curvature_test_g2_senses = []
    if use_curvature_prior:
        point_to_adjoining_gradients = defaultdict(list)
        for index,point1,point2 in all_gradient_tests():
            # this is not, though at first glance it appears to be, a symmetric adjacency matrix
            # it's adjacency of points to segments, which are of different types
            point_to_adjoining_gradients[point1].append((index,1)) # (index, sense):  gradients are computed from p1 to p2
            point_to_adjoining_gradients[point2].append((index,-1))# reading gradient *from* p2 we later need to multiply by -1
            
        for adjoining_gradients in point_to_adjoining_gradients.values():
            for (ind1,sense1),(ind2,sense2) in combinations(adjoining_gradients,2):
                curvature_test_g1_indices.append(ind1)
                curvature_test_g2_indices.append(ind2)
                curvature_test_g1_senses.append(sense1)
                curvature_test_g2_senses.append(sense2)
        
        del point_to_adjoining_gradients

    all_points_arrays = [np.array(s) for s in all_points_sets]
    del all_points_sets,point_to_type
    decoupled_graph = decoupled_graph.tocsr()
    gradient_test_p1_types = np.array(gradient_test_p1_types,dtype=np.ubyte)
    gradient_test_p1_indices = np.array(gradient_test_p1_indices,dtype=np.longlong)
    gradient_test_p2_types = np.array(gradient_test_p2_types,dtype=np.ubyte)
    gradient_test_p2_indices = np.array(gradient_test_p2_indices,dtype=np.longlong)
    gradient_test_distances = np.array(gradient_test_distances)
    num_gradient_tests = len(gradient_test_distances)
    
    curvature_test_distances = (gradient_test_distances[curvature_test_g1_indices]+gradient_test_distances[curvature_test_g2_indices])/2
    curvature_test_g1_indices = np.array(curvature_test_g1_indices,dtype=np.longlong)
    curvature_test_g2_indices = np.array(curvature_test_g2_indices,dtype=np.longlong)
    curvature_test_g1_senses = np.array(curvature_test_g1_senses,dtype=np.byte)
    curvature_test_g2_senses = np.array(curvature_test_g2_senses,dtype=np.byte)
    
    mean_estimated_segment_length = gradient_test_distances.mean()
    print_callback(f"Minimum estimated segment length {gradient_test_distances.min():.2f}")
    print_callback(f"Mean estimated segment length: {mean_estimated_segment_length:.2f}")
            
    def init_guess_decoupled_zs():
        '''Distance weighted average of simple drape zs of boundary points'''
        decoupled_weighted_z_sum = np.zeros(num_decoupled_points)
        decoupled_weight_sum = np.zeros(num_decoupled_points)

        for pt in decoupled_group_boundary_points:
            pt_type,pt_index = pt
            pt_x,pt_y = all_points_arrays[pt_type][pt_index]
            boundary_pt_z = float(terrain_interpolator(np.array([[pt_x,pt_y]]))[0]) #  could do these all together if performance ever a problem
            ind = decoupled_graph_index(pt_type,pt_index)
            assert ind >= num_decoupled_points
            # n.b. if performance is ever an issue the method below can take multiple indices at once
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
            
        return decoupled_weighted_z_sum/decoupled_weight_sum    
            
    # Define priors - both implemented by hand for speed and compatibility with autodiff
    
    print_callback(f"Spatial mismatch prior scale is {mismatch_prior_std}")
    
    # Exponential grade prior
    grade_scale = np.tan(slope_prior_scale*np.pi/180)
    print_callback(f"Slope prior scale of {slope_prior_scale}\N{DEGREE SIGN} gives grade of {grade_scale*100:.1f}%")
    grade_exp_dist_lambda = 1/grade_scale
    
    # Normalized grade_pdfs used to give physical interpretation to slope_continuity_scale, though not used in main optimization
    def slope_continuity_param_from_scale(scale):
        return 1/2/scale**2
    def grade_pdf_normalized(slope_continuity_scale,grade_scale,grade):
        slope_continuity_param = slope_continuity_param_from_scale(slope_continuity_scale)
        norm_const = (np.exp(1/(4*slope_continuity_param*grade_scale**2))*(np.pi**0.5) * erfc(1/(2*slope_continuity_param**0.5*grade_scale)))/(2*slope_continuity_param**0.5)
        return norm_const**-1*np.exp(-grade/grade_scale-slope_continuity_param*grade**2)
    
    def grade_pdf2_normalized(grade_scale,grade):
        norm_const = grade_scale
        return norm_const**-1*np.exp(-grade/grade_scale)
    
    if slope_continuity_scale==None:
        if slope_continuity_desired_impact==1:
            print_callback("Desired impact of slope continuity is probability multiplier of 1; setting slope continuity scale to infinity")
            slope_continuity_scale = np.inf
        else:
            print_callback(f"Attempting to compute slope continuity prior scale based on desired probability multiplier (for grade of 0.5) of {slope_continuity_desired_impact} (this is experimental and may fail)")
            def continuity_prior_optim_target(slope_continuity_scale):
                impact_sc_grade05 = grade_pdf_normalized(slope_continuity_scale,grade_scale,0.5)/grade_pdf2_normalized(grade_scale,0.5)
                return (impact_sc_grade05-slope_continuity_desired_impact)**2
            cpsmax = 2**0.5 # we get NaN if we go lower
            scresult = minimize(continuity_prior_optim_target,np.zeros(1)+slope_continuity_scale_default,bounds=Bounds(-np.inf,cpsmax)) 
            print_callback (f"Continuity prior solved with status: {scresult['message']}")
            slope_continuity_scale = float(scresult["x"])

    slope_continuity_param = slope_continuity_param_from_scale(slope_continuity_scale)
    
    if slope_continuity_param>0:
        print_callback(f"Slope continuity prior scale (expressed as grade) is {slope_continuity_scale*100:.1f}%")
        impact_sc_grade0 = grade_pdf_normalized(slope_continuity_scale,grade_scale,0)/grade_pdf2_normalized(grade_scale,0)
        impact_sc_grade05 = grade_pdf_normalized(slope_continuity_scale,grade_scale,0.5)/grade_pdf2_normalized(grade_scale,0.5)
        print (f"Probability multiplier attributable to slope continuity prior for grade=0.0: {impact_sc_grade0}")
        print (f"Probability multiplier attributable to slope continuity prior for grade=0.5: {impact_sc_grade05}")
    
    def grade_logpdf(grade): # not normalized, for efficiency+precision in main optimization
        return -grade_exp_dist_lambda*grade - slope_continuity_param*grade**2 

    # 2d Gaussian offset prior
    sigma = mismatch_prior_std
    sigma_sq = sigma**2
    def squareoffset_logpdf(x): # not normalized, for efficiency+precision in main optimization
        return -(x/sigma_sq)/2 
            
    # Pareto curvature prior
    curvature_loc = -curvature_scale
    assert curvature_scale>0
    assert curvature_shape>0
    def curvature_logpdf(c):
        x = (c-curvature_loc)/curvature_scale 
        return -(curvature_shape+1) * torch.log(x)
        
    # Define posterior log likelihood

    # Prepare arrays for likelihood tests
    gradient_test_weights = np_to_torch(gradient_test_distances / mean_estimated_segment_length)
    gradient_test_inv_distances = np_to_torch(gradient_test_distances**-1)
    gradient_test_p1_indices = np_to_torch(gradient_test_p1_indices)
    gradient_test_p2_indices = np_to_torch(gradient_test_p2_indices)
    gradient_test_p1_types = np_to_torch(gradient_test_p1_types)
    gradient_test_p2_types = np_to_torch(gradient_test_p2_types)
    curvature_test_g1_indices = np_to_torch(curvature_test_g1_indices)
    curvature_test_g1_senses = np_to_torch(curvature_test_g1_senses)
    curvature_test_g2_indices = np_to_torch(curvature_test_g2_indices)
    curvature_test_g2_senses = np_to_torch(curvature_test_g2_senses)
    curvature_test_distances = np_to_torch(curvature_test_distances)
    
    del gradient_test_distances
    all_points_arrays = [np_to_torch(a) for a in all_points_arrays]

    if len(all_points_arrays[FIXED]):
        fixed_zs = terrain_interpolator(all_points_arrays[FIXED])
    else:
        fixed_zs = torch.zeros(0,dtype=torch.double)
    
    def unpack_opt_params(opt_params):
        if not torch.is_tensor(opt_params): # may be the case if called from something other than the optimizer
            opt_params = np_to_torch(opt_params)
        assert len(opt_params)==2*num_estimated_points+num_decoupled_points
        point_offsets = torch.reshape(opt_params[0:num_estimated_points*2],all_points_arrays[ESTIMATED].shape)
        decoupled_zs = opt_params[num_estimated_points*2:]
        return point_offsets,decoupled_zs
        
    def init_opt_params():
        point_offsets = np_to_torch(np.zeros((num_estimated_points*2),float))
        decoupled_zs = np_to_torch(init_guess_decoupled_zs())
        return torch.cat((point_offsets,decoupled_zs))
        
    def pack_optim_bounds(max_offset_dist):
        offset_bounds_lower = np.zeros(num_estimated_points*2)-max_offset_dist
        offset_bounds_upper = np.zeros(num_estimated_points*2)+max_offset_dist
        decoupled_bounds_lower = np.zeros(num_decoupled_points)+terrain_min_height
        decoupled_bounds_upper = np.zeros(num_decoupled_points)+terrain_max_height
        return np.concatenate((offset_bounds_lower,decoupled_bounds_lower)),np.concatenate((offset_bounds_upper,decoupled_bounds_upper))
    
    def minus_log_likelihood(opt_params):
        point_offsets,decoupled_zs = unpack_opt_params(opt_params)
        
        estimated_zs = terrain_interpolator(all_points_arrays[ESTIMATED] + point_offsets)
        
        n1_zs = torch.zeros(num_gradient_tests,dtype=torch.double)
        n2_zs = torch.zeros(num_gradient_tests,dtype=torch.double)
        n1_zs[gradient_test_p1_types==FIXED] = fixed_zs[gradient_test_p1_indices[gradient_test_p1_types==FIXED]] # fixme any faster to precompute these index arrays? possibly in polymorphic tensor
        n2_zs[gradient_test_p2_types==FIXED] = fixed_zs[gradient_test_p2_indices[gradient_test_p2_types==FIXED]]
        n1_zs[gradient_test_p1_types==ESTIMATED] = estimated_zs[gradient_test_p1_indices[gradient_test_p1_types==ESTIMATED]]
        n2_zs[gradient_test_p2_types==ESTIMATED] = estimated_zs[gradient_test_p2_indices[gradient_test_p2_types==ESTIMATED]]
        n1_zs[gradient_test_p1_types==DECOUPLED] = decoupled_zs[gradient_test_p1_indices[gradient_test_p1_types==DECOUPLED]]
        n2_zs[gradient_test_p2_types==DECOUPLED] = decoupled_zs[gradient_test_p2_indices[gradient_test_p2_types==DECOUPLED]]
        
        neighbour_heightdiffs = n2_zs-n1_zs
        neighbour_grades = neighbour_heightdiffs*gradient_test_inv_distances
        neighbour_likelihood = (grade_logpdf(abs(neighbour_grades))*gradient_test_weights).sum()
        
        curvature_likelihood = 0
        if use_curvature_prior:
            # gradients*sense gives gradient looking out from the pair midpoint
            # to assess slope continuity we need to invert one of these gradients again to simulate arriving and leaving
            curvature_g1s = neighbour_grades[curvature_test_g1_indices]*curvature_test_g1_senses*-1
            curvature_g2s = neighbour_grades[curvature_test_g2_indices]*curvature_test_g2_senses
            curvatures = grade_change(curvature_g1s,curvature_g2s)
            curvature_likelihood = (curvature_logpdf(curvatures)).sum() # fixme if we stick with this remove curvature_test_distances and rename curvature to angles or something
        
        # Log likelihood of point offsets
        offset_square_distances = ((point_offsets**2).sum(axis=1))
        offset_likelihood = squareoffset_logpdf(offset_square_distances).sum()
        print (f"NL {float(neighbour_likelihood)} OL {float(offset_likelihood)} CL {float(curvature_likelihood)}")

        return -(neighbour_likelihood + offset_likelihood + curvature_likelihood)

    def minus_log_likelihood_gradient(opt_params):
        if not torch.is_tensor(opt_params):
            opt_params = np_to_torch(opt_params)
        opt_params.requires_grad = True
        minus_log_likelihood(opt_params).backward() 
        return opt_params.grad
        
    def reconstruct_geometries(opt_results):
        final_offsets,final_decoupled_zs = unpack_opt_params(opt_results)
        final_log_likelihood = -minus_log_likelihood(opt_results)
        final_estimated_zs = terrain_interpolator(all_points_arrays[ESTIMATED] + final_offsets)
        
        # Print report on offsets
        offset_distances = ((final_offsets**2).sum(axis=1))**0.5
        mean_offset_dist = float(offset_distances.mean())
        max_offset_dist = float(offset_distances.max())
        print_callback (f"Offset distance mean={mean_offset_dist:.2f}, max={max_offset_dist:.2f}")

        # Interpolate decoupled points: iterate through decoupled_group_boundary_points, as they are likely fewer in number than the decoupled points themselves

        def get_point_output(index_tuple):
            pt_type,pt_index = index_tuple
            x,y = all_points_arrays[pt_type][pt_index]
            if pt_type==ESTIMATED:
                return x,y,float(final_estimated_zs[pt_index])
            elif pt_type==FIXED:
                return x,y,float(fixed_zs[pt_index])
            elif pt_type==DECOUPLED:
                return x,y,final_decoupled_zs[pt_index]
            else: assert False

        return [LineString([get_point_output(pt_index) for pt_index in row_point_indices]) for row_point_indices in point_indices_all_rows]
    
    return_dict = dict(grade_logpdf=grade_logpdf,
                  squareoffset_logpdf=squareoffset_logpdf,
                  minus_log_likelihood=lambda x: float(minus_log_likelihood(x)),
                  minus_log_likelihood_gradient=minus_log_likelihood_gradient,
                  initial_guess=init_opt_params(),
                  reconstruct_geometries_from_optimizer_results=reconstruct_geometries,
                  mismatch_prior_std=mismatch_prior_std,
                  optim_bounds = pack_optim_bounds
                  )
    BayesianDrapeModel = namedtuple("BayesianDrapeModel",return_dict)
    return BayesianDrapeModel(**return_dict)
    
def fit_model(model,maxiter,max_offset_dist=np.inf,print_callback=print):
    initial_log_likelihood = -model.minus_log_likelihood(model.initial_guess)
    last_ll = initial_log_likelihood
    callback_count = 0
    
    def callback(x):
        nonlocal last_ll,callback_count
        callback_count += 1
        ll = float(-model.minus_log_likelihood(x))
        lldiff = abs(ll-last_ll)
        last_ll = ll
        text = f"Iteration {callback_count} log likelihood = {ll:.1f} (-{lldiff:.3f})          "
        #print_callback (f+"\r",end="")
        print_callback(text)
    
    lower_bounds,upper_bounds = model.optim_bounds(max_offset_dist)
    print_callback (f"Starting optimizer log likelihood = {-model.minus_log_likelihood(model.initial_guess):.1f}")
    result = minimize(model.minus_log_likelihood,model.initial_guess,callback = callback,bounds=Bounds(lower_bounds,upper_bounds),jac=model.minus_log_likelihood_gradient,options=dict(maxiter=maxiter)) 
    print_callback (f"\nOptimizer terminated with status: {result['message']}")
    
    optimizer_results = result["x"]
    print_callback ("Reconstructing geometries")
    return [LineString(geom) for geom in model.reconstruct_geometries_from_optimizer_results(optimizer_results)]

def fit_model_from_command_line_options():
    from optparse import OptionParser
    import rioxarray # gdal raster is another option 
    import pandas as pd
    import geopandas as gp
    
    maxiter_default = 1000
    
    op = OptionParser()
    op.add_option("--TERRAIN-INPUT",dest="terrainfile",help="[REQUIRED] Terrain model",metavar="FILE")
    op.add_option("--POLYLINE-INPUT",dest="shapefile",help="[REQUIRED] Polyline feature class e.g. network or GPS trace",metavar="FILE")
    op.add_option("--OUTPUT",dest="outfile",help="[REQUIRED] Output feature class",metavar="FILE")
    op.add_option("--SLOPE-PRIOR-SCALE",dest="slope_prior_scale",help=f"Scale of exponential prior for path slope (equivalent to mean slope; defaults to {slope_prior_scale_default})",metavar="ANGLE_IN_DEGREES",type="float")
    op.add_option("--SPATIAL-MISMATCH-PRIOR-STD",dest="mismatch_prior_std",help="Standard deviation of zero-centred Gaussian prior for spatial mismatch (in spatial units of projection; defaults to half terrain raster cell size)",metavar="DISTANCE",type="float")
    op.add_option("--SPATIAL-MISMATCH-MAX",dest="mismatch_max",help="Maximum permissible spatial mismatch (in spatial units of projection; defaults to 4x mismatch prior std)",metavar="DISTANCE",type="float")
    op.add_option("--SLOPE-CONTINUITY-PRIOR-SCALE",dest="slope_continuity",help=f"Slope continuity prior scale parameter (defaults to {slope_continuity_scale_default})",metavar="GRADE",type="float")
    op.add_option("--SLOPE-CONTINUITY-PRIOR-DESIRED-IMPACT",dest="slope_continuity_desired_impact",help=f"Compute slope continuity prior according to desired multiplier on probability of GRADE=0.5",metavar="MULTIPLIER",type="float")
    op.add_option("--CURVATURE-PRIOR-SCALE",dest="curvature_scale",help=f"Curvature prior scale (defaults to {curvature_scale_default})",metavar="GRADE_CHANGE_PER_UNIT_LENGTH",type="float")
    op.add_option("--CURVATURE-PRIOR-SHAPE",dest="curvature_shape",help=f"Curvature prior shape (defaults to {curvature_shape_default})",metavar="PARETO_SHAPE_PARAM",type="float")
    op.add_option("--USE-CURVATURE-PRIOR",dest="test_curvature",action="store_true",help="Enable curvature prior",default=False)
    op.add_option("--GPU",dest="cuda",action="store_true",help="Enable GPU acceleration")
    op.add_option("--SIMPLE-DRAPE-FIELD",dest="simpledrapefield",help="Instead of estimating heights, perform ordinary drape of features over terrain where FIELDNAME=true",metavar="FIELDNAME")
    op.add_option("--DECOUPLE-FIELD",dest="decouplefield",help="Instead of estimating heights, decouple features from terrain where FIELDNAME=true (useful for bridges/tunnels)",metavar="FIELDNAME")
    op.add_option("--MAXITER",dest="maxiter",help=f"Maximum number of optimizer iterations (defaults to {maxiter_default})",metavar="N",type="int",default=maxiter_default)
    (options,args) = op.parse_args()
    
    if options.cuda and not torch.cuda.is_available():
            op.error("PyTorch CUDA is not available; try running without --GPU")
    
    if options.slope_continuity and options.slope_continuity_desired_impact:
        op.error('Cannot run with conflicting options: --SLOPE-CONTINUITY-DESIRED-IMPACT and --SLOPE-CONTINUITY-PARAM')
    
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
                        slope_continuity_scale = options.slope_continuity,
                        slope_continuity_desired_impact = options.slope_continuity_desired_impact,
                        use_curvature_prior = options.test_curvature,
                        curvature_scale = options.curvature_scale,
                        curvature_shape = options.curvature_shape,
                        simpledraped_geometries_mask = simpledrape_geometries_mask,
                        decoupled_geometries_mask = decouple_geometries_mask,
                        use_cuda = options.cuda)
    
    del terrain_xs,terrain_ys,terrain_data,terrain_raster
    
    if not options.mismatch_max:
        options.mismatch_max = model.mismatch_prior_std * 4
    
    net_df.geometry = fit_model(model,options.maxiter,options.mismatch_max)

    print (f"Writing output to {options.outfile}") 
    net_df.to_file(options.outfile)

if __name__ == "__main__":
    fit_model_from_command_line_options()
    