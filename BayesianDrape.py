# -*- coding: utf-8 -*-

"""
***************************************************************************
By Crispin Cooper
Copyright Cardiff University 2021                                         
Licensed under GNU Affero v3
***************************************************************************
"""

from scipy.optimize import minimize,Bounds
from scipy.special import erfc
import scipy.sparse
from scipy.sparse import lil_matrix
from terrain_interpolator import TerrainInterpolator
import numpy as np
from ordered_set import OrderedSet
from itertools import tee,combinations,groupby
import operator
import shapely.wkb
from shapely.geometry import LineString,MultiLineString,MultiPoint,Point
import torch
from collections import namedtuple,defaultdict
import time

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def remove_consecutive_duplicates(iterable):
    # based on unique_justseen in the itertools docs
    return list(map(next, map(operator.itemgetter(1), groupby(iterable))))

def grade_change_angle(g1,g2):
    return abs(torch.atan(g1)-torch.atan(g2))/np.pi*180

def insert_vertices(line,splitpoints,tolerance):
    '''insert new vertices on a line as close as possible to each splitpoint;
    ignore new vertices if closer than tolerance to existing ones'''
    existing_points = [(line.project(Point(xy)),xy) for xy in line.coords]
    potential_cuts = sorted([(line.project(p),p) for p in splitpoints])
    
    # discard cuts close to previous cuts
    cuts = []
    last_cut_dist = 0
    for c in potential_cuts:
        cut_dist,p = c
        if cut_dist-last_cut_dist>tolerance:
            cuts.append(c)
            last_cut_dist = cut_dist
        
    newcoords = []
    while existing_points: 
        next_pt_dist,next_pt = existing_points[0]
        if not cuts:
            # if no cuts left, output remaining points and end
            # note that conversely if no points are left, remaining cuts are discarded
            existing_points.pop(0)
            newcoords.append(next_pt)
        else:
            next_cut_dist,next_cut = cuts[0]
            if abs(next_pt_dist-next_cut_dist)<tolerance:
                cuts.pop(0) # discard cuts close to existing points
            elif next_pt_dist<next_cut_dist:
                existing_points.pop(0)
                newcoords.append(next_pt)
            else:
                cuts.pop(0)
                newcoords.append(next_cut)
    return LineString(newcoords)

def insert_points_on_gridlines(line,grid,tolerance):
    # shapely does weird things if line has z!
    # especially on lines with vertical segments, even if they're not an intersection
    # so, lines where we want to keep z should NOT be passed to this function
    assert not grid.has_z
    assert not line.has_z 
    
    intersection = line.intersection(grid) 
    assert intersection.geom_type in ['MultiPoint', 'Point', 'LineString', 'GeometryCollection', 'MultiLineString']
    if not intersection.is_empty:
        # flatten to all points
        if intersection.geom_type in ['GeometryCollection','MultiPoint','MultiLineString']:
            intersection = intersection.geoms 
        else:
            intersection = [intersection] # treat as a collection of 1 item
        point_only_intersection = []
        for item in intersection:
            if item.geom_type=='Point':
                point_only_intersection.append(item)
            elif item.geom_type=='LineString':
                point_only_intersection += [Point(c) for c in item.coords]
            else:
                print (item.geom_type)
                assert False # intersection between lines is not point or linestring
        return insert_vertices(line,point_only_intersection,tolerance),len(point_only_intersection)
    else:
        return line,0
        
z_error_prior_scale_default = 0.25  # 1.0 gives behaviour of Hutchinson (1996)
slope_continuity_param_default = 0.5
slope_prior_mean_default = 2.66

nugget_default_as_fraction_of_cell_size = 1/100
new_vertex_tolerance_as_fraction_of_cell_size = 1/100
gradient_smooth_window_default_as_fraction_of_cell_size = 1/100

def build_model(terrain_index_xs,terrain_index_ys,terrain_zs,
                geometries,
                slope_prior_mean=slope_prior_mean_default,z_error_prior_scale=None,slope_continuity_param=None,
                nugget=None,
                z_smooth_window=0,gradient_smooth_window=None,
                fix_geometries_mask=None,decoupled_geometries_mask=None,
                use_cuda=False,
                print_callback=print):


    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        device = torch.device("cuda")
        def np_to_torch(x):
            if torch.is_tensor(x):
                assert x.is_cuda
                return x
            else:
                return torch.from_numpy(x).cuda()
    else:
        device = torch.device("cpu")
        def np_to_torch(x):
            if torch.is_tensor(x):
                return x
            else:
                return torch.from_numpy(x)

    # computed parameter defaults
    if fix_geometries_mask is None:
        fix_geometries_mask = [0]*len(geometries)
    if decoupled_geometries_mask is None:
        decoupled_geometries_mask = [0]*len(geometries)
    
    cellstepx = terrain_index_xs[1]-terrain_index_xs[0]
    cellstepy = terrain_index_ys[1]-terrain_index_ys[0]
    cellsizex = abs(cellstepx)
    cellsizey = abs(cellstepy)
    if nugget is None:
        nugget = min(cellsizex,cellsizey)*nugget_default_as_fraction_of_cell_size
    if gradient_smooth_window is None:
        gradient_smooth_window = min(cellsizex,cellsizey)*gradient_smooth_window_default_as_fraction_of_cell_size
        print_callback(f"Gradient smooth window set from cell size: {gradient_smooth_window}")
        
    # constant parameter defaults 
    # (can't use default arguments as we want to support caller passing None)
    if slope_prior_mean is None:
        slope_prior_mean = slope_prior_mean_default
    if slope_continuity_param is None:
        slope_continuity_param = slope_continuity_param_default
    if z_error_prior_scale is None:
        z_error_prior_scale = z_error_prior_scale_default
    
    assert slope_continuity_param <= 1
    assert slope_continuity_param >= 0
    
    if z_error_prior_scale==0:
        z_smooth_window=0
        
    new_vertex_tolerance = min(cellsizex,cellsizey)*new_vertex_tolerance_as_fraction_of_cell_size

    # ensure terrain has positively incrementing axes to keep interpolator happy
    if cellstepx<0:
        terrain_index_xs = np.flip(terrain_index_xs)
        terrain_zs = np.flip(terrain_zs,axis=1)
    if cellstepy<0:
        terrain_index_ys = np.flip(terrain_index_ys)
        terrain_zs = np.flip(terrain_zs,axis=0)
        
    # change from yx to xy ordering
    terrain_zs = terrain_zs.T
    
    def regularly_spaced(x):
        xd = np.diff(x)
        return np.all(np.abs(xd-xd[0])<xd[0]/100000)
        
    assert regularly_spaced(terrain_index_xs)
    assert regularly_spaced(terrain_index_ys)
    
    terrain_xs = np_to_torch(terrain_index_xs.copy())
    terrain_ys = np_to_torch(terrain_index_ys.copy())
    
    terrain_data = np_to_torch(terrain_zs.copy())
    terrain_min_height = float(terrain_data.min())
    terrain_max_height = float(terrain_data.max())
    print_callback(f"Terrain height range from {terrain_min_height:.2f} to {terrain_max_height:.2f}")
    terrain_interpolator_internal = TerrainInterpolator(terrain_xs,terrain_ys, terrain_data)
    del terrain_data
    
    def terrain_interpolator(points_to_interpolate):
        # calling contiguous() silences the performance warning from PyTorch
        # swapping dimensions on our definition of points throughout, and ensuring all tensors are created contiguous, gives maybe 10% speed boost - reverting for now as that's premature
        if not torch.is_tensor(points_to_interpolate):
            points_to_interpolate = np_to_torch(points_to_interpolate)
        return terrain_interpolator_internal.forward(points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous(),z_smooth_window)    

    def get_max_tile_DZs(points_to_interpolate):
        if not torch.is_tensor(points_to_interpolate):
            points_to_interpolate = np_to_torch(points_to_interpolate)
        return terrain_interpolator_internal.get_max_tile_DZs(points_to_interpolate[:,0].contiguous(),points_to_interpolate[:,1].contiguous(),gradient_smooth_window)    

    # Feature type model: decoupled takes precedence over fixed as people are likely to fix large flat areas and decouples to smaller features like bridges within them
    num_point_types = 3
    ESTIMATED,FIXED,DECOUPLED = range(num_point_types)
    def get_feature_type(decoupled,fixed):
        if decoupled:
            return DECOUPLED
        elif fixed:
            return FIXED
        else:
            return ESTIMATED

    # Add points on lines wherever they cross terrain model grid cells 
    print_callback ("Inserting extra vertices on polylines to interpolate terrain")
    xmin = terrain_xs.min()
    xmax = terrain_xs.max()
    ymin = terrain_ys.min()
    ymax = terrain_ys.max()
    x_gridlines = [((x,ymin),(x,ymax)) for x in terrain_xs]
    y_gridlines = [((xmin,y),(xmax,y)) for y in terrain_ys]
    del terrain_xs, terrain_ys
    gridlines = MultiLineString(x_gridlines+y_gridlines)
    del x_gridlines,y_gridlines
    inserted_vertex_count = 0
    geometries_split = []
    for geom,fix,decouple in zip(geometries,fix_geometries_mask,decoupled_geometries_mask): 
        assert type(geom)==LineString
        use_input_z = get_feature_type(decouple,fix)==FIXED
        if not use_input_z: 
            geom = LineString(shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2)).coords) # remove z coordinates *before* removing duplicates
        geom = LineString(remove_consecutive_duplicates(geom.coords))
        if not use_input_z:
            geom, num_new_vertices = insert_points_on_gridlines(geom,gridlines,new_vertex_tolerance) # to ensure every terrain point is interpolated
            inserted_vertex_count += num_new_vertices
        geometries_split.append(geom)
    del gridlines
    geometries = geometries_split
    print_callback (f"  ({inserted_vertex_count} extra vertices added)")
    
    # Build point model: we represent each linestring with a point index list 
    # point (x,y)s are initially stored in OrderedSets so we can give the same index to matching line endpoints; these OrderedSets are later converted to arrays
    
    all_points_sets = [OrderedSet(),OrderedSet(),OrderedSet()]
    assert len(all_points_sets)==num_point_types
    
    # Point type model (controls behaviour of optimizer at each point)

    # in contrast to the feature type model above, precedence order for endpoints (the only ones with potential conflict) is fixed, estimated, decoupled. We keep these in a dict to keep track of conflicts
    # n.b. if this precedence is ever changed, will need to change handling of fixed links during "inserting extra vertices" phase above
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

    print_callback ("Building spatial model")
    # first pass through data to resolve point types
    for geom,decoupled,fixed in zip(geometries,decoupled_geometries_mask,fix_geometries_mask):
        xs,ys = geom.coords.xy
        feature_type = get_feature_type(decoupled,fixed)
        for x,y in zip(xs,ys):
            set_point_type((x,y),feature_type)
    del decoupled_geometries_mask,fix_geometries_mask
    
    fixed_zs = {}
    
    # second pass through data to create point model
    point_indices_all_rows = []
    for geom in geometries:
        assert geom.geom_type=='LineString'
        point_indices = []
        if geom.has_z:
            coords = list(geom.coords)
        else:
            coords = [(x,y,0.) for (x,y) in geom.coords]
        for x,y,z in coords:
            point_type = point_to_type[(x,y)]
            index = all_points_sets[point_type].add((x,y))
            point_indices.append((point_type,index))
            if point_type==FIXED:
                fixed_zs[index]=z
        point_indices_all_rows.append(point_indices)

    fixed_zs = np_to_torch(np.array([fixed_zs[i] for i in range(len(all_points_sets[FIXED]))]))
    
    num_estimated_points = len(all_points_sets[ESTIMATED])
    num_decoupled_points = len(all_points_sets[DECOUPLED])
    num_fixed_points = len(all_points_sets[FIXED])
    total_num_points = num_estimated_points+num_decoupled_points+num_fixed_points
    print_callback (total_num_points,"points in total")
    print_callback (num_estimated_points,"points for estimation")
    print_callback (num_decoupled_points,"points are decoupled")
    print_callback (num_fixed_points,"points are fixed")
    assert num_estimated_points > 0
    
    est_pts_total_adjoining_segment_length = np.zeros(num_estimated_points)

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
                assert (type1,index1)!=(type2,index2) # duplicate point
                yield (type1,index1,point1),(type2,index2,point2)
    
    # Build all matrices (another pass through data)
    # provided we optimize all points together, we only store distances in one direction
    # otherwise each gradient likelihood is counted twice, which breaks log likelihood
    # if we did ever want to compute log likelihood for a single point, we would need a symmetric distance matrix

    for (type1,index1,(x1,y1)),(type2,index2,(x2,y2)) in all_neighbouring_points():
        dist = ((x2-x1)**2+(y2-y1)**2)**0.5
        assert dist>0
        
        if type1==ESTIMATED:
            est_pts_total_adjoining_segment_length[index1] += dist
        if type2==ESTIMATED:
            est_pts_total_adjoining_segment_length[index2] += dist
        
        if (type1,type2)!=(FIXED,FIXED):
            add_gradient_test(type1,index1,type2,index2,dist)
            
        if type1==DECOUPLED:
            decoupled_graph_add(index1,type2,index2,dist)
        elif type2==DECOUPLED:
            assert type1!=DECOUPLED # handled above
            decoupled_graph_add(index2,type1,index1,dist)
    
    all_points_arrays = [np.array(s) for s in all_points_sets]
    del all_points_sets,point_to_type
    decoupled_graph = decoupled_graph.tocsr()
    gradient_test_p1_types = np.array(gradient_test_p1_types,dtype=np.ubyte)
    gradient_test_p1_indices = np.array(gradient_test_p1_indices,dtype=np.longlong)
    gradient_test_p2_types = np.array(gradient_test_p2_types,dtype=np.ubyte)
    gradient_test_p2_indices = np.array(gradient_test_p2_indices,dtype=np.longlong)
    gradient_test_distances = np.array(gradient_test_distances)
    num_gradient_tests = len(gradient_test_distances)
    
    print_callback(f"Minimum estimated segment length {gradient_test_distances.min():.2f}")
            
    def init_guess_decoupled_zs():
        '''Inverse distance weighted average of simple drape zs of boundary points'''
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
    
    print_callback(f"Z error prior scale is {z_error_prior_scale}")
    
    # Exponential*Normal grade prior
    assert slope_prior_mean < 90
    grade_mean = np.tan(slope_prior_mean*np.pi/180) 
    print_callback(f"Slope prior mean of {slope_prior_mean}\N{DEGREE SIGN} gives grade of {grade_mean*100:.1f}%")
    
    def halfnormal_logpdf_param_from_mean(mean):
        assert mean>0
        return 1/(mean**2*np.pi)
    def exp_logpdf_param_from_mean(mean):
        assert mean>0
        return 1/mean
    
    grade_exp_dist_param = exp_logpdf_param_from_mean(grade_mean)*(1-slope_continuity_param)
    grade_halfnormal_dist_param = halfnormal_logpdf_param_from_mean(grade_mean)*slope_continuity_param
    
    # Grade prior is a mixture of exponential and Gaussian parameterized by slope_continuity_param
    def grade_logpdf(grade): # not normalized, for efficiency+precision in main optimization
        return -grade_exp_dist_param*grade - grade_halfnormal_dist_param*grade**2 

    # vertical error prior: Gaussian with std = DZ / sqrt(12) * z_error_prior_scale
    # So logpdf = -0.5 x**2 / std**2 = -0.5 x**2 / ((DZ**2 * z_error_prior_scale**2)/12) = -6/z_error_prior_scale**2  x**2 / DZ**2
    if z_error_prior_scale>0:
        k = 6 / z_error_prior_scale**2
        def z_error_logpdf(vertex_z_error,tile_max_DZ_inverse_square):
            res = -k * tile_max_DZ_inverse_square * vertex_z_error**2 
            res[torch.isnan(res)] = 0 # zero z-error on a flat tile is fine
            return res
    else:
        # allows setting z_error_prior_scale==0 to get ordinary drape
        def z_error_logpdf(vertex_z_error,tile_max_DZ_inverse_square):
            res = torch.zeros(vertex_z_error.shape,device=device)
            res[vertex_z_error!=0]=-np.inf
            return res
        
    # Define posterior log likelihood

    # Prepare arrays for likelihood tests
    gradient_test_weights = np_to_torch(gradient_test_distances)
    z_error_weights = np_to_torch(est_pts_total_adjoining_segment_length / 2)
    gradient_test_inv_distances = np_to_torch(gradient_test_distances**-1)
    gradient_test_p1_indices = np_to_torch(gradient_test_p1_indices)
    gradient_test_p2_indices = np_to_torch(gradient_test_p2_indices)
    gradient_test_p1_types = np_to_torch(gradient_test_p1_types)
    gradient_test_p2_types = np_to_torch(gradient_test_p2_types)
    
    del gradient_test_distances, est_pts_total_adjoining_segment_length
    all_points_arrays = [np_to_torch(a) for a in all_points_arrays]
    
    def unpack_opt_params(opt_params):
        opt_params = np_to_torch(opt_params)
        assert len(opt_params)==num_estimated_points+num_decoupled_points
        z_errors = opt_params[0:num_estimated_points]
        decoupled_zs = opt_params[num_estimated_points:]
        return z_errors,decoupled_zs
        
    def init_opt_params():
        z_errors = np.zeros(num_estimated_points,float)
        decoupled_zs = init_guess_decoupled_zs()
        return np.concatenate((z_errors,decoupled_zs),axis=None)

    def pack_optim_bounds():
        terrain_DZ = terrain_max_height - terrain_min_height
        z_error_bounds_lower = np.zeros(num_estimated_points)-terrain_DZ
        z_error_bounds_upper = np.zeros(num_estimated_points)+terrain_DZ
        decoupled_bounds_lower = np.zeros(num_decoupled_points)+terrain_min_height
        decoupled_bounds_upper = np.zeros(num_decoupled_points)+terrain_max_height
        return np.concatenate((z_error_bounds_lower,decoupled_bounds_lower)),np.concatenate((z_error_bounds_upper,decoupled_bounds_upper))
    
    z1s_mask_fixed = gradient_test_p1_types==FIXED
    z2s_mask_fixed = gradient_test_p2_types==FIXED
    z1s_mask_est = gradient_test_p1_types==ESTIMATED
    z2s_mask_est = gradient_test_p2_types==ESTIMATED
    z1s_mask_decoupled = gradient_test_p1_types==DECOUPLED
    z2s_mask_decoupled = gradient_test_p2_types==DECOUPLED
    fixed_zs_p1_indices = gradient_test_p1_indices[z1s_mask_fixed]
    fixed_zs_p2_indices = gradient_test_p2_indices[z2s_mask_fixed]
    est_zs_p1_indices = gradient_test_p1_indices[z1s_mask_est]
    est_zs_p2_indices = gradient_test_p2_indices[z2s_mask_est]
    decoupled_zs_p1_indices = gradient_test_p1_indices[z1s_mask_decoupled]
    decoupled_zs_p2_indices = gradient_test_p2_indices[z2s_mask_decoupled]
    
    interpolated_zs = terrain_interpolator(all_points_arrays[ESTIMATED])
    estimated_point_max_tile_DZs_inverse_sq = (get_max_tile_DZs(all_points_arrays[ESTIMATED])+nugget)**-2
            
    def likelihood_breakdown(opt_params):
        z_errors,decoupled_zs = unpack_opt_params(opt_params)
        
        estimated_zs = interpolated_zs + z_errors
        
        z1s = torch.zeros(num_gradient_tests,dtype=torch.double,device=device)
        z2s = torch.zeros(num_gradient_tests,dtype=torch.double,device=device)
        z1s[z1s_mask_fixed] = fixed_zs[fixed_zs_p1_indices] 
        z2s[z2s_mask_fixed] = fixed_zs[fixed_zs_p2_indices]
        z1s[z1s_mask_est] = estimated_zs[est_zs_p1_indices]
        z2s[z2s_mask_est] = estimated_zs[est_zs_p2_indices]
        z1s[z1s_mask_decoupled] = decoupled_zs[decoupled_zs_p1_indices]
        z2s[z2s_mask_decoupled] = decoupled_zs[decoupled_zs_p2_indices]
        
        neighbour_heightdiffs = z2s-z1s
        neighbour_grades = neighbour_heightdiffs*gradient_test_inv_distances
        neighbour_likelihood = (grade_logpdf(abs(neighbour_grades))*gradient_test_weights).sum()
        
        # Log likelihood of z errors
        z_error_likelihood = (z_error_logpdf(z_errors,estimated_point_max_tile_DZs_inverse_sq)*z_error_weights).sum()
        return neighbour_likelihood,z_error_likelihood
        
    def likelihood_report(opt_params):
        n,o = map(float,likelihood_breakdown(opt_params))
        return n+o,f"   (Z error likelihood {o:.1f}, Slope likelihood {n:.1f})"
        
    def minus_log_likelihood(opt_params):
        n,o = likelihood_breakdown(opt_params)
        return -(n+o)

    def mll_cpu(opt_params):
        return minus_log_likelihood(opt_params).cpu()

    def minus_log_likelihood_gradient(opt_params):
        opt_params = np_to_torch(opt_params)
        opt_params.requires_grad = True
        minus_log_likelihood(opt_params).backward()
        return torch.nan_to_num(opt_params.grad,nan=0,posinf=np.inf,neginf=-np.inf).cpu()
        
    def reconstruct_geometries(opt_results):
        final_z_errors,final_decoupled_zs = unpack_opt_params(opt_results)
        final_log_likelihood = -minus_log_likelihood(opt_results)
        final_estimated_zs = interpolated_zs + final_z_errors
        
        # Print report on z errors
        mean_z_error = float(final_z_errors.mean())
        std_z_error = float(final_z_errors.std())
        max_z_error = float(final_z_errors.max())
        min_z_error = float(final_z_errors.min())
        print_callback (f"Z error mean={mean_z_error:.2f}, std={std_z_error:.2f}, range=({min_z_error:.2f},{max_z_error:.2f})")

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
                  z_error_logpdf=z_error_logpdf,
                  minus_log_likelihood=lambda x: float(mll_cpu(x)),
                  minus_log_likelihood_gradient=minus_log_likelihood_gradient,
                  likelihood_report=likelihood_report,
                  initial_guess=init_opt_params(),
                  reconstruct_geometries_from_optimizer_results=reconstruct_geometries,
                  z_error_prior_scale=z_error_prior_scale,
                  optim_bounds = pack_optim_bounds
                  )
    BayesianDrapeModel = namedtuple("BayesianDrapeModel",return_dict)
    return BayesianDrapeModel(**return_dict)
    
def fit_model(model,maxiter,print_callback=print,reportiter=5,differentiate=True,returnStats=False):
    initial_log_likelihood,initial_lik_report = model.likelihood_report(model.initial_guess)
    print_callback(f"Num params: {model.initial_guess.shape[0]}")
    last_ll = initial_log_likelihood
    last_time = time.perf_counter()
    callback_count = 0
    
    last_params = model.initial_guess
    second_params = model.initial_guess
    penultimate_params = model.initial_guess
    
    def callback(x):
        nonlocal last_ll,last_time,callback_count,last_params,penultimate_params,second_params
        penultimate_params = last_params
        last_params = x.copy()
        if callback_count==0:
            second_params = x.copy()
        callback_count += 1
        if reportiter>0 and callback_count%reportiter==0:
            ll = float(-model.minus_log_likelihood(x))
            lldiff = abs(ll-last_ll)
            last_ll = ll
            t = time.perf_counter()
            tdiff = t - last_time
            last_time = t
            text = f"Iteration {callback_count} log likelihood = {ll:.1f} (-{lldiff:.3f} over {reportiter} iterations) ({tdiff/reportiter*100:.2f}s per 100 iterations)         "
            print_callback (text+"\r",end="")
    
    lower_bounds,upper_bounds = model.optim_bounds()
    jac = model.minus_log_likelihood_gradient if differentiate else None
    print_callback (f"Starting optimizer log likelihood = {initial_log_likelihood:.1f}\n{initial_lik_report}")
    init_time = time.perf_counter()
    if maxiter>0:
        result = minimize(model.minus_log_likelihood,model.initial_guess,callback = callback,bounds=Bounds(lower_bounds,upper_bounds),jac=jac,options=dict(maxiter=maxiter,maxfun=maxiter*20)) 
    else:
        result = {"message":"no iteration","x":model.initial_guess}
    end_time = time.perf_counter()
    t_secs = end_time - init_time
    
    print_callback (f"\nOptimizer terminated with status: {result['message']}")
    print_callback (f"Total time: {t_secs:.3g} seconds")
    optimizer_results = result["x"]
    end_log_likelihood,end_lik_report = model.likelihood_report(optimizer_results)
    print_callback (f"Final optimizer log likelihood after {callback_count} iterations = {end_log_likelihood:.1f}\n{end_lik_report}")
    
    second_ll = float(-model.minus_log_likelihood(second_params))
    penultimate_ll = float(-model.minus_log_likelihood(penultimate_params))
    print_callback(f"First step change in log likelihood: {second_ll-initial_log_likelihood}")
    print_callback(f"First step mean change in param: {abs(model.initial_guess-second_params).mean()}")
    print_callback(f"Final step change in log likelihood: {end_log_likelihood-penultimate_ll}")
    print_callback(f"Final step mean change in param: {abs(penultimate_params-optimizer_results).mean()}")
    print_callback(f"Total change in log likelihood: {end_log_likelihood-initial_log_likelihood}")
    print_callback(f"Total mean change in param: {abs(model.initial_guess-optimizer_results).mean()}")
    
    print_callback ("Reconstructing geometries")
    res = [LineString(geom) for geom in model.reconstruct_geometries_from_optimizer_results(optimizer_results)]
    
    if not returnStats:
        return res
    else:
        return res, optimizer_results, callback_count

def fit_model_from_command_line_options():
    from optparse import OptionParser
    import rioxarray # gdal raster is another option 
    import pandas as pd
    import geopandas as gp
    from pyproj.crs import CRS
    
    maxiter_default = 20000
    
    op = OptionParser()
    op.add_option("--TERRAIN-INPUT",dest="terrainfile",help="[REQUIRED] Terrain model",metavar="FILE")
    op.add_option("--POLYLINE-INPUT",dest="shapefile",help="[REQUIRED] Polyline feature class e.g. network or GPS trace",metavar="FILE")
    op.add_option("--OUTPUT",dest="outfile",help="[REQUIRED] Output feature class",metavar="FILE")
    op.add_option("--FIX-FIELD",dest="fixfield",help="Instead of estimating heights, preserve input z on features where FIELDNAME=true",metavar="FIELDNAME")
    op.add_option("--DECOUPLE-FIELD",dest="decouplefield",help="Instead of estimating heights, decouple features from terrain where FIELDNAME=true (useful for bridges/tunnels)",metavar="FIELDNAME")
    
    op.add_option("--Z-ERROR-PRIOR-SCALE",dest="z_error_prior_scale",help=f"Scale of Gaussian prior for z mismatch (Defaults to {z_error_prior_scale_default}. 1.0 gives Hutchinson (1996) model. Set to 0 for simple drape. Higher numbers allow greater z correction)",metavar="SCALE",type="float")
    op.add_option("--SLOPE-PRIOR-MEAN",dest="slope_prior_mean",help=f"Mean of prior for path slope (defaults to {slope_prior_mean_default}; measured in degrees but prior is over grade)",metavar="ANGLE_IN_DEGREES",type="float")
    op.add_option("--SLOPE-CONTINUITY-PARAM",dest="slope_continuity_param",help=f"Parameter for shape of slope prior; set to 0 for exponential or 1 for Gaussian; defaults to {slope_continuity_param_default}.",metavar="PARAM",type="float")
    op.add_option("--MAXITER",dest="maxiter",help=f"Maximum number of optimizer iterations (defaults to {maxiter_default}, set to 0 for naive drape)",metavar="N",type="int",default=maxiter_default)
    op.add_option("--NUGGET",dest="nugget",help=f"Nugget / assumed minimum elevation difference in flat terrain cell (defaults to {nugget_default_as_fraction_of_cell_size}*cell size)",metavar="Z_DISTANCE",type="float")
    op.add_option("--GPU",dest="cuda",action="store_true",help="Enable GPU acceleration")
    op.add_option("--NUM-THREADS",dest="threads",help="Set number of threads for multiprocessing (defaults to number of available cores)",type="int",metavar="N")
    op.add_option("--IGNORE-PROJ-MISMATCH",dest="ignore_proj_mismatch",action="store_true",help="Ignore mismatched projections",default=False)
    op.add_option("--ITERATION-REPORT-EVERY",dest="reportiter",help="Report log likelihood every N iterations (set to 0 for never)",type="int",metavar="N",default=5)
    op.add_option("--DISABLE-AUTODIFF",dest="disable_autodiff",action="store_true",help="Disable automatic differentiation (slow!)",default=False)
    
    (options,args) = op.parse_args()

    if options.threads:
        torch.set_num_threads(options.threads)    

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

    net_crs = net_df.crs
    terr_crs = CRS(terrain_raster.rio.crs)
    if not net_crs.equals(terr_crs,True):
        print(f"Coordinate reference systems:\n  Polyline CRS: {net_crs.name}\n  Terrain CRS:  {terr_crs.name}")
        if not options.ignore_proj_mismatch:
            op.error("Coordinate reference systems of polylines and terrain do not appear to match. Reproject, fix CRS metadata or - if you think you know better - use --IGNORE-PROJ-MISMATCH at your peril.")
        else:
            print("Ignoring mismatched projections, don't say I didn't warn you!")
    else:
        print(f"Using {net_crs.name}")
    
    terrain_xs = np.array(terrain_raster.x,np.float64)
    terrain_ys = np.array(terrain_raster.y,np.float64)
    terrain_data = terrain_raster.data[0]
    
    if options.fixfield and options.decouplefield and np.logical_and(net_df[options.fixfield],net_df[options.decouplefield]).any():
        print ("Warning: Input contains features marked both as fixed and decoupled. These will be treated as decoupled.")
        
    fix_geometries_mask = net_df[options.fixfield] if options.fixfield else None
    decouple_geometries_mask = net_df[options.decouplefield] if options.decouplefield else None
        
    model = build_model(terrain_xs,terrain_ys,terrain_data,net_df.geometry,
                        slope_prior_mean = options.slope_prior_mean,
                        z_error_prior_scale = options.z_error_prior_scale,
                        slope_continuity_param = options.slope_continuity_param,
                        nugget = options.nugget,
                        fix_geometries_mask = fix_geometries_mask,
                        decoupled_geometries_mask = decouple_geometries_mask,
                        use_cuda = options.cuda)
    
    del terrain_xs,terrain_ys,terrain_data,terrain_raster
    
    net_df.geometry = fit_model(model,options.maxiter,
                                reportiter=options.maxiter,differentiate=(not options.disable_autodiff))

    print (f"Writing output to {options.outfile}") 
    net_df.to_file(options.outfile)

if __name__ == "__main__":
    fit_model_from_command_line_options()
    
