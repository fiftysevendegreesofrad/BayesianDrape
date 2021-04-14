import BayesianDrape

grade_scale = np.tan(2.2*np.pi/180)
def my_exp_logpdf(grade):
    return -np.log(grade_exp_dist_lambda) - grade_exp_dist_lambda*grade

def test_priors():
    pass
    
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
        
    