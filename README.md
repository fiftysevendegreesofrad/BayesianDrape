# BayesianDrape
Drape GPS traces and path networks over terrain using Bayesian maximum likelihood estimation

A bit like Kriging, but with a network, and the underlying data is the elevation model which we assume to be correct just lacking sufficient detail.
https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12979

Also a bit like Kalman filtering, but making use of extra information specific to the task of draping paths over terrain.

As a physical analog you might like to think about a network made of rubber bands being stretched to fit over the terrain. This is not very accurate as (1) rubber bands only stretch over convex surfaces while this algorithm stretches over concave surfaces also; (2) the rubber band metaphor implies the ends of the rubber band are at a fixed position, whereas here, nothing is fixed. A better, though still approximate physical analog might be to imagine the rubber bands being made of statically charged clingfilm, with a tendency to stick to the surface regardless of whether concave or convex, balanced by an opposing stiffness such that they resist excessive bending. But (unless we allow vertical errors) they will never leave the surface of the terrain, they just have a little local flexibility in regard to exactly which part of it they stick to. This is as far as the analogy can go, because we do not really simulate the elastic and electrostatic forces governing our imaginary material. What we compute is a local maximum a posteriori likelihood for the error model, but given that caveat, visual thinkers may still prefer to imagine stiff clingfilm rubber bands.

Splitting to 10m segments draped on 50m raster gives distribution of (hg+hl)/len for prior (although this includes outliers it should still be representative enough)
Maximum:	0.509859
Sum:	15044.436881
Mean:	0.042823
Standard Deviation:	0.039105

Actually we should check normality and remove outliers if needed. exported to ec_per_len.txt.

also do it by angle
Minimum:	0
Maximum:	27.0152
Sum:	859474.466267
Mean:	2.446443
Standard Deviation:	2.221301

Will need to check for precision issues in sum likelihoods.

If it comes to analytical differentiation, do I do it per point (assuming others don't move) or for the ensemble (e.g. automatically with autograd pyautodiff auto-diff pytorch)?
intro to autodiff suitable for jax: https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf
https://github.com/google/jax
looks great! basically says supports numpy so let's get started with numpy sparse.
comparison of some autograds: https://arxiv.org/ftp/arxiv/papers/1606/1606.06311.pdf
http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/


profile visualisation
https://jiffyclub.github.io/snakeviz/
python -m cProfile -s cumtime BayesianDrape.py  --JUST-LLTEST >tmp && head -n 60 tmp

idea for sparseness when i come back to it https://www.google.com/search?client=firefox-b-d&q=scipy+sparse+adjacency+matrix+distance