# BayesianDrape
Drape GPS traces and path networks over terrain using Bayesian maximum likelihood estimation.

![image](https://github.com/fiftysevendegreesofrad/BayesianDrape/assets/12543309/917089f8-92d9-44b7-adef-7d561e56f18c)

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

You can create a naive drape for comparison by setting MAXITER=0:

    python BayesianDrape.py --TERRAIN-INPUT data/all_os50_terrain.tif --POLYLINE-INPUT data/test_awkward_link.shp --OUTPUT naive_drape_output.shp --MAXITER=0

BayesianDrape is tested on `.tiff` and `.shp` files, but should be able to use any raster supported by [rioxarray](https://corteva.github.io/rioxarray/stable/) and any vector supported by [Geopandas](https://geopandas.org/).

For more options:

    python BayesianDrape.py --help
    
        Options:
          -h, --help            show this help message and exit
          --TERRAIN-INPUT=FILE  [REQUIRED] Terrain model
          --POLYLINE-INPUT=FILE
                                [REQUIRED] Polyline feature class e.g. network or GPS trace
          --OUTPUT=FILE         [REQUIRED] Output feature class
          --FIX-FIELD=FIELDNAME
                                Instead of estimating heights, preserve input z on features where FIELDNAME=true
          --DECOUPLE-FIELD=FIELDNAME
                                Instead of estimating heights, decouple features from terrain where FIELDNAME=true (useful for bridges/tunnels)
          --Z-ERROR-PRIOR-SCALE=SCALE
                                Scale of Gaussian prior for z mismatch (Defaults to 0.25. 1.0 gives Hutchinson (1996) model. Set to 0 for simple drape. Higher numbers allow greater z correction)
          --SLOPE-PRIOR-MEAN=ANGLE_IN_DEGREES
                                Mean of prior for path slope (defaults to 2.66; measured in degrees but prior is over grade)
          --SLOPE-CONTINUITY-PARAM=PARAM
                                Parameter for shape of slope prior; set to 0 for exponential or 1 for Gaussian; defaults to 0.5.
          --PITCH-ANGLE-PRIOR-MEAN=ANGLE_IN_DEGREES
                                Pitch angle prior mean (defaults to inf)
          --MAXITER=N           Maximum number of optimizer iterations (defaults to 20000, set to 0 for naive drape)
          --NUGGET=Z_DISTANCE   Nugget / assumed minimum elevation difference in flat terrain cell (defaults to 0.01*cell size)
          --GPU                 Enable GPU acceleration
          --NUM-THREADS=N       Set number of threads for multiprocessing (defaults to number of available cores)
          --IGNORE-PROJ-MISMATCH
                                Ignore mismatched projections
          --ITERATION-REPORT-EVERY=N
                                Report log likelihood every N iterations (set to 0 for never)
          --DISABLE-AUTODIFF    Disable automatic differentiation (slow!)

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
