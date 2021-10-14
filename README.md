# BayesianDrape
Drape GPS traces and path networks over terrain using Bayesian maximum likelihood estimation

## Setup

Download
    git clone https://github.com/fiftysevendegreesofrad/BayesianDrape.git
    cd BayesianDrape

Install the minimal environment needed to use BayesianDrape:
    conda env create -f conda_env_configs/conda_env_minimal.yml
(I also provide conda_env_current.yml which contains a working MPI + Jupyter version, but is not needed for basic use).

## Use

A minimal example: 
    python BayesianDrape.py --TERRAIN-INPUT data/all_os50_terrain.tif --POLYLINE-INPUT data/test_awkward_link.shp --OUTPUT testout.shp 

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
