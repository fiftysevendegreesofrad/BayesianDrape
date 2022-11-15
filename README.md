# BayesianDrape
Drape GPS traces and path networks over terrain using Bayesian maximum likelihood estimation

![image](https://user-images.githubusercontent.com/12543309/137376777-69e42b05-269a-4144-8bf0-16fcf1cd355c.png)

## Setup

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/index.html) if you don't already have it.

2. Download BayesianDrape

        git clone https://github.com/fiftysevendegreesofrad/BayesianDrape.git
        
3. Install the minimal environment needed to use BayesianDrape:

        conda env create -f BayesianDrape/conda_env_configs/conda_env_minimal.yml

   (I also provide `conda_env_current.yml` which contains a working MPI + Jupyter version, but is not needed for basic use).

## Use

A minimal example: 

    conda activate bayesiandrape
    cd BayesianDrape
    python BayesianDrape.py --TERRAIN-INPUT data/all_os50_terrain.tif --POLYLINE-INPUT data/test_awkward_link.shp --OUTPUT bayes_drape_output.shp 

You can create a naive drape for comparison by setting the spatial mismatch prior scale to zero:

    python BayesianDrape.py --TERRAIN-INPUT data/all_os50_terrain.tif --POLYLINE-INPUT data/test_awkward_link.shp --OUTPUT naive_drape_output.shp --SPATIAL-MISMATCH-PRIOR-SCALE=0

BayesianDrape is tested on `.tiff` and `.shp` files, but should be able to use any raster supported by [rioxarray](https://corteva.github.io/rioxarray/stable/) and any vector supported by [Geopandas](https://geopandas.org/).

For more options:

    python BayesianDrape.py --help
    
    Options:
      -h, --help            show this help message and exit
      --TERRAIN-INPUT=FILE  [REQUIRED] Terrain model
      --POLYLINE-INPUT=FILE
                            [REQUIRED] Polyline feature class e.g. network or GPS
                            trace
      --OUTPUT=FILE         [REQUIRED] Output feature class
      --FIX-FIELD=FIELDNAME
                            Instead of estimating heights, preserve input z on
                            features where FIELDNAME=true
      --DECOUPLE-FIELD=FIELDNAME
                            Instead of estimating heights, decouple features from
                            terrain where FIELDNAME=true (useful for
                            bridges/tunnels)
      --SPATIAL-MISMATCH-PRIOR-SCALE=DISTANCE
                            Standard deviation of zero-centred Gaussian prior for
                            spatial mismatch (in spatial units of projection;
                            defaults to half terrain raster cell size)
      --SLOPE-PRIOR-SCALE=ANGLE_IN_DEGREES
                            Scale of exponential prior for path slope (equivalent
                            to mean slope; defaults to 2.66; measured in degrees
                            but prior is over grade)
      --SLOPE-CONTINUITY-PRIOR-SCALE=ANGLE_IN_DEGREES
                            Scale of normal (slope continuity) prior for path
                            slope (equivalent to mean slope; defaults to 90;
                            measured in degrees but prior is over grade)
      --PITCH-ANGLE-PRIOR-SCALE=ANGLE_IN_DEGREES
                            Pitch angle prior scale (defaults to 1.28)
      --SPATIAL-MISMATCH-MAX=DISTANCE
                            Maximum permissible spatial mismatch (in spatial units
                            of projection; defaults to maximum terrain tile
                            dimension)
      --MAXITER=N           Maximum number of optimizer iterations (defaults to
                            10000)
      --GPU                 Enable GPU acceleration
      --NUM-THREADS=N       Set number of threads for multiprocessing (defaults to
                            number of available cores)
      --IGNORE-PROJ-MISMATCH
                            Ignore mismatched projections

## Use as a library

You can `import BayesianDrape` into your own code. 

    import BayesianDrape
    import geopandas as gp
    import numpy as np
    import rioxarray

    # Read network with geopandas - alternatively just create your own list of shapely geometries
    net_df = gp.read_file("my_network_polylines.shp")
    shapely_geometries = net_df.geometry

    # Read terrain with rioxarray - alternatively create your own 1-d numpy arrays for x/y coordinates, and a 2-d array of z values
    terrain_raster = rioxarray.open_rasterio("my_terrain.tif")
    terrain_xs = np.array(terrain_raster.x,np.float64)
    terrain_ys = np.array(terrain_raster.y,np.float64)
    terrain_data = terrain_raster.data[0]
        
    # Build and fit model
    model = BayesianDrape.build_model(terrain_xs,terrain_ys,terrain_data,shapely_geometries)
    max_iterations = 10000
    draped_shapely_geometries = BayesianDrape.fit_model(model,max_iterations)
    
Further arguments to `build_model()` allow for changing priors, fixed and decoupled points, etc. For the definitive usage example, look at the bottom of `BayesianDrape.py` to see what it does if called from the command line.
