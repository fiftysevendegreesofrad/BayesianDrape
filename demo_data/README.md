This is a set of example data suitable for demonstrating the BayesianDrape code.

`input/terrain1.tif` OS Open Terrain 50 terrain model

`input/test_net.{shp,dbf,shx,prj,cpg}` matching Open Street Map road network

The output examples are derived by running BayesianDrape as shown in the README file with the code.

`output_examples/bayesian/testoutbayes.{shp,dbf,shx,prj,cpg}` Bayesian drape output

`output_examples/bayesian/testout.{shp,dbf,shx,prj,cpg}` Naive drape output obtained by running BayesianDrape with --MAXITER=0

The best way to view these outputs is to load into a 3d shapefile viewer such as ArcScene, and set vertical exaggeration somewhere between 5 and 10.